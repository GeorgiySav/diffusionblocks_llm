#include <cstdio>

#include "llm/llama_sampler.h"

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  try {
    llm::llama_generate::run_generate_cli(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "generate failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
