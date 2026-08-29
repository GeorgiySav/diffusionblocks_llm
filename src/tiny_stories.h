#pragma once

#include <string>
#include <fstream>

#include <nn/data/token_dataset.h>
#include <nn/data/dataloader.h>

const std::string kTinyStoriesTrainFilePath = "data/TinyStories/TinyStories-train.txt";
const std::string kTinyStoriesValidFilePath = "data/TinyStories/TinyStories-valid.txt";

inline constexpr std::string format_story(const std::string& story) {
  return "<|start_story|>" + story.substr(1, story.length() - 2) + "<|end_story|>";
}

// returns a list of stories
// the .txt file has stories separated by a <|endoftext|> token, but also have newlines
inline std::vector<std::string> load_tiny_stories(const std::string& file_path = kTinyStoriesValidFilePath) {
  std::ifstream file(file_path);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + file_path);
  }
  
  std::vector<std::string> stories;  

  std::string story;
  while (std::getline(file, story, '<')) {
    std::string endoftext;
    if (std::getline(file, endoftext, '>')) {
      if (endoftext == "|endoftext|") {
        stories.push_back(story);
        story.clear();
      } else {
        story += "<" + endoftext + ">";
      }
    } else {
      story += "<" + endoftext;
    }
  }
  if (!story.empty()) {
    stories.push_back(story);
  }

  file.close();
  return stories;
}