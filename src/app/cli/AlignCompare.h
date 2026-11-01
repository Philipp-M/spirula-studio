#pragma once

#include <filesystem>
#include <vector>

namespace spirula::align {
void compare_runs(const std::filesystem::path& output,const std::vector<std::filesystem::path>& runs);
}
