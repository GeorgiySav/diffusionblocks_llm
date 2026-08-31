#include <cstdio>

#include "llm/sampler.h"

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // flush progress even when piped

  try {
    llm::generate::run_generate(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "generate failed: %s\n", e.what());
    return 1;
  }

  return 0;
}
