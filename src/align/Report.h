#pragma once

#include <filesystem>
#include <string>

namespace spirula::align {
void write_report(const std::filesystem::path& output, const std::string& manifest);
}
