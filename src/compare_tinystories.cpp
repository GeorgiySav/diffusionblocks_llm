#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "llm/sampler.h"
#include "llm/trainer.h"
#include "train_config.h"

// Memory-fair comparison: GPT, DiffusionRecurrent and DiffusionBlocks all get
// the same width (n_embed/n_heads/mlp_hidden_dim), context length, and batch
// size, and -- crucially -- the same number of transformer layers *active in
// the compute graph on any single training step* (kActiveDepth): GPT's
// n_layers, DiffusionRecurrent's prelude+core+coda, and DiffusionBlocks'
// layers-per-block are all kActiveDepth. That equalizes per-step activation
// memory (the thing that actually bounds how deep a network you can train)
// across all three.
//
// DiffusionBlocks alone then gets to spend that same per-step budget kMulti
// times over: its *total* network is kActiveDepth * kMulti distinct layers,
// split into kMulti independently-trained blocks, of which only one
// (kActiveDepth layers) is ever in memory at once. If it reaches a lower
// validation loss than GPT under this constraint, that's the paper's central
// claim demonstrated directly: more effective capacity for the same
// per-step memory footprint.
//
// Caveat worth keeping in mind reading the results: both diffusion models
// process a 2T-length sequence per active layer (the clean+noisy
// concatenation trick), roughly doubling per-layer compute/memory over
// GPT's T-length sequence -- that's inherent to the training scheme itself,
// not a knob this experiment is trying to equalize away.
static constexpr int kActiveDepth = 4;
// DiffusionBlocks: kActiveDepth * kMulti total layers, one of kMulti blocks
// sampled per step. Lowered from 4: at kMulti=4 over a ~2150-step budget,
// each block only saw ~537 updates (~34K examples) and produced fragmented,
// barely-grammatical text even with more reverse-ODE sampling steps at
// generation time -- confirmed undertraining, not a sampling-schedule
// issue. kMulti=2 halves total capacity but doubles updates per block
// within the same time budget.
static constexpr int kMulti = 2;

// At most kMaxSeconds of wall-clock time per model. max_steps below is sized
// per model kind from measured throughput (~0.30 s/step GPT, ~0.75 s/step
// for either diffusion model, TF32 + the sync-deferral/RoPE-caching changes)
// so the LR schedule's cosine decay actually spans the whole budget instead
// of only reaching the tail of a schedule sized for many more steps than
// will run; max_seconds is a safety net in case throughput is worse than
// estimated, not the primary stopping mechanism.
static constexpr double kMaxSeconds = 1800.0;

static TrainConfig build_cfg(const char* name, ModelKind kind) {
  TrainConfig cfg;
  cfg.model = kind;
  // A 20M-token slice (~4% of the full corpus) instead of the whole thing --
  // see kTinyStoriesSmallTrainTokens's comment -- so several epochs fit
  // inside the time budget below instead of a sliver of one.
  cfg.train_tokens_path = llm::data::kTinyStoriesSmallTrainTokens;
  cfg.valid_tokens_path = llm::data::kTinyStoriesValidTokens;

  cfg.vocab_size = 4096;
  cfg.n_embed = 256;
  cfg.n_heads = 8;
  cfg.mlp_hidden_dim = 4 * cfg.n_embed;
  cfg.dropout = 0.1f;
  cfg.context_length = 256;

  cfg.n_layers = (kind == ModelKind::DiffusionBlocks) ? kActiveDepth * kMulti : kActiveDepth;
  cfg.n_diffusion_blocks = kMulti;
  cfg.n_prelude_layers = 1;
  cfg.n_coda_layers = kActiveDepth - 2; // + core's 1 layer == kActiveDepth
  cfg.sigma_data = 1.0f / std::sqrt(float(cfg.n_embed));

  cfg.batch_size = 64;
  // Peak activation memory scales with micro_batch_size, not batch_size (see
  // run_training_loop's gradient-accumulation comment) -- at micro_batch_size
  // == batch_size == 64, the 2T=512-length attention score tensors across
  // kActiveDepth=4 active layers overrun a 16GB GPU for either diffusion
  // model (confirmed: a genuine cudaErrorMemoryAllocation, not a bug), so
  // they accumulate over 2 micro-batches of 32 instead. GPT has no 2T
  // doubling and was only using ~6GB of the ~14.6GB free at micro_batch_size
  // 32, so it goes straight to 64 (no accumulation at all) -- same effective
  // batch_size, less overhead, better GPU utilization.
  cfg.micro_batch_size = (kind == ModelKind::GPT) ? 64 : 32;
  cfg.max_seconds = kMaxSeconds;

  // GPT runs ~2.5x faster per step than either diffusion model (no 2T
  // sequence, no EDM noise sampling/weighting) -- see run()'s comment above.
  // Sized from a measured run at kMaxSeconds (GPT: 5316 steps actually
  // completed in 1800s -> 0.339 s/step; DiffusionRecurrent: 2153 steps ->
  // 0.836 s/step), rounded down slightly so the cosine schedule fully
  // anneals to min_lr comfortably inside the time budget instead of being
  // cut off ~10% short of its own end by max_seconds.
  cfg.max_steps = (kind == ModelKind::GPT) ? 5300 : 2150;
  cfg.warmup_steps = cfg.max_steps / 40; // ~2.5%
  cfg.log_interval = cfg.max_steps / 60;
  cfg.eval_interval = cfg.max_steps / 4; // 4 evals across the run
  cfg.eval_iters = 20;
  cfg.checkpoint_interval = cfg.eval_interval;
  // Persistent location (not cleaned up on exit) so a checkpoint survives
  // for generation afterward -- each run() call is typically a separate
  // process invocation (see main()'s per-architecture selector), so only
  // this model's own stale checkpoint is cleared, not its siblings'.
  cfg.checkpoint_path = std::string("checkpoints/quick_compare/") + name + ".ckpt";
  return cfg;
}

// Sample right from cfg -- the checkpoint's actual architecture -- instead
// of through generate.exe's CLI, whose default TrainConfig doesn't match
// this experiment's n_layers/n_prelude_layers/n_coda_layers overrides.
// num_sampling_steps is a generation-time-only knob (diffusion /
// diffusion_blocks): raising it costs seconds, not minutes, so it's worth
// trying before spending more GPU time on additional training.
static void sample(const char* name, ModelKind kind, const TrainConfig& cfg,
                    int num_sampling_steps = 16) {
  std::printf("\n--- %s sample (num_sampling_steps=%d) ---\n", name, num_sampling_steps);
  const std::string encoding = "data/TinyStories/tokenizer-4096.json";
  const std::string prompt = "Once upon a time";
  constexpr int kMaxNewTokens = 150;
  constexpr int kTopK = 40;
  switch (kind) {
    case ModelKind::GPT:
      llm::generate::detail::run_generate_gpt(cfg, cfg.checkpoint_path, prompt, encoding,
                                               kMaxNewTokens, kTopK);
      break;
    case ModelKind::DiffusionRecurrent:
      llm::generate::detail::run_generate_diffusion(cfg, cfg.checkpoint_path, prompt, encoding,
                                                     kMaxNewTokens, kTopK, num_sampling_steps);
      break;
    case ModelKind::DiffusionBlocks:
      llm::generate::detail::run_generate_diffusion_blocks(cfg, cfg.checkpoint_path, prompt, encoding,
                                                            kMaxNewTokens, kTopK, num_sampling_steps);
      break;
  }
}

static void run(const char* name, ModelKind kind) {
  std::printf("\n========== %s ==========\n", name);
  TrainConfig cfg = build_cfg(name, kind);
  std::filesystem::remove(cfg.checkpoint_path);
  llm::train::run_training(cfg);
  sample(name, kind, cfg);
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0); // see output before a crash, not just after
  // Pass "gpt" / "diffusion" / "diffusion_blocks" to train (+ sample) just
  // one -- useful for isolating a crash to a specific architecture instead
  // of waiting through all three every time. Pass "regen_<name> <steps>" to
  // skip training and resample an existing checkpoint with a different
  // num_sampling_steps -- e.g. "regen_diffusion_blocks 50".
  const std::string only = (argc > 1) ? argv[1] : "";

  try {
    if (only.rfind("regen_", 0) == 0) {
      const std::string name = only.substr(6);
      const ModelKind kind = model_kind_from_string(name == "diffusion_recurrent" ? "diffusion" : name);
      const int steps = (argc > 2) ? std::atoi(argv[2]) : 16;
      TrainConfig cfg = build_cfg(name.c_str(), kind);
      sample(name.c_str(), kind, cfg, steps);
      return 0;
    }

    if (only.empty() || only == "gpt") run("gpt", ModelKind::GPT);
    if (only.empty() || only == "diffusion") run("diffusion_recurrent", ModelKind::DiffusionRecurrent);
    if (only.empty() || only == "diffusion_blocks") run("diffusion_blocks", ModelKind::DiffusionBlocks);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "comparison failed: %s\n", e.what());
    return 1;
  }

  std::printf("\n=== done -- compare final val loss / val raw ce above ===\n"
              "checkpoints kept at checkpoints/quick_compare/<name>.ckpt\n");
  return 0;
}
