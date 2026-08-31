#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm {

// Encodes/decodes text by shelling out to scripts/tokenizer_cli.py.
// `encoding` is either a tiktoken encoding name (e.g. "gpt2") or a path to a
// custom tokenizer.json trained by scripts/train_tokenizer.py -- the CLI
// tells them apart by whether the string names an existing file.
class Tokenizer {
 public:
  explicit Tokenizer(std::string encoding, std::string python = "py")
      : encoding_(std::move(encoding)), python_(std::move(python)) {}

  std::vector<int32_t> encode(const std::string& text) const {
    const std::filesystem::path in_path = make_temp_path();
    write_file(in_path, text);
    const std::string out = run_cli("encode", in_path);
    std::filesystem::remove(in_path);

    std::vector<int32_t> ids;
    std::istringstream iss(out);
    std::string line;
    while (std::getline(iss, line)) {
      if (!line.empty()) ids.push_back(std::stoi(line));
    }
    return ids;
  }

  std::string decode(const std::vector<int32_t>& ids) const {
    std::ostringstream body;
    for (int32_t id : ids) body << id << '\n';

    const std::filesystem::path in_path = make_temp_path();
    write_file(in_path, body.str());
    const std::string out = run_cli("decode", in_path);
    std::filesystem::remove(in_path);
    return out;
  }

 private:
  static std::filesystem::path make_temp_path() {
    static std::atomic<uint64_t> counter{0};
    return std::filesystem::temp_directory_path() /
           ("tiktoken_cli_" + std::to_string(counter.fetch_add(1)) + ".txt");
  }

  static void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Tokenizer: failed to write " + path.string());
    f << content;
  }

  std::string run_cli(const std::string& mode, const std::filesystem::path& in_path) const {
    const std::string cmd = python_ + " scripts/tokenizer_cli.py " + mode +
                             " --input \"" + in_path.string() + "\"" +
                             " --encoding \"" + encoding_ + "\" 2>&1";

#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "rb");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) throw std::runtime_error("Tokenizer: failed to start: " + cmd);

    std::string out;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) out.append(buf, n);

#ifdef _WIN32
    const int status = _pclose(pipe);
#else
    const int status = pclose(pipe);
#endif
    if (status != 0) {
      throw std::runtime_error("Tokenizer: `" + cmd + "` failed:\n" + out);
    }
    return out;
  }

  std::string encoding_;
  std::string python_;
};

}  // namespace llm
