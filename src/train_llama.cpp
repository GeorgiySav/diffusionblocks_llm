#include <cstdio>

#include "llama_config.h"
#include "llm/llama_trainer.h"

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  try {
    llm::llama_train::run_training(LlamaTrainConfig{});
  } catch (const std::exception& e) {
    std::fprintf(stderr, "training failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
