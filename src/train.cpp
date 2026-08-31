#include <cstdio>

#include "llm/trainer.h"
#include "train_config.h"

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // flush progress even when piped

  try {
    TrainConfig cfg;
    if (argc > 1) cfg.model = model_kind_from_string(argv[1]); // "gpt", "diffusion", or "diffusion_blocks"
    llm::train::run_training(cfg);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "training failed: %s\n", e.what());
    return 1;
  }

  return 0;
}
