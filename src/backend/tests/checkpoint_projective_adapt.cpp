#include "checkpoint/Adapt.h"
#include "core/CheckpointIO.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_floats(std::ostream& out, const std::string& name,
                  const std::vector<float>& values) {
    const std::string header = ckpt::npy_header("<f4", values.size());
    const size_t size = header.size() + values.size() * sizeof(float);
    ckpt::tar_header(out, name + ".npy", size);
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    ckpt::tar_pad(out, size);
}

std::vector<float> read_floats(const std::filesystem::path& path, const std::string& name) {
    std::ifstream in(path / "state.tar", std::ios::binary);
    for (const auto& member : ckpt::tar_index(in)) {
        if (member.name != name + ".npy") continue;
        const auto info = ckpt::npy_locate(in, member.data_offset, member.size);
        std::vector<float> values(info.data_bytes / sizeof(float));
        in.seekg(info.data_offset);
        in.read(reinterpret_cast<char*>(values.data()), info.data_bytes);
        return values;
    }
    throw std::runtime_error("missing adapted buffer: " + name);
}

void check_capacity(const std::filesystem::path& root, int capacity,
                    const std::vector<int>& keep) {
    ckpt::TargetLayout target;
    target.max_num_splats = capacity;
    const auto output = root / std::to_string(capacity);
    if (!ckpt::adapt_checkpoint(root, target, output))
        throw std::runtime_error("checkpoint was not adapted");
    for (const auto& [name, channels] :
         std::vector<std::pair<std::string, int>>{{"eng.projective_state", 4},
                                                 {"eng.projective_information", 1}}) {
        const auto actual = read_floats(output, name);
        if (actual.size() != static_cast<size_t>(capacity * channels))
            throw std::runtime_error(name + ": adapted capacity mismatch");
        for (int row = 0; row < capacity; ++row) {
            for (int channel = 0; channel < channels; ++channel) {
                const float expected = row < static_cast<int>(keep.size())
                    ? 10 * (keep[row] + 1) + channel : 0;
                if (actual[row * channels + channel] != expected)
                    throw std::runtime_error(name + ": incorrect retained or padded row");
            }
        }
    }
}

}

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() /
        ("ss-checkpoint-projective-adapt-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(root);
        {
            std::ofstream out(root / "state.tar", std::ios::binary);
            const std::string state =
                "{\"max_num_splats\": 4, \"cur_num_splats\": 4, \"num_sh\": 0}";
            ckpt::tar_write_bytes(out, "state.json", state.data(), state.size());
            write_floats(out, "world.opacities", {-2, 1, -1, 2});
            write_floats(out, "eng.projective_state",
                         {10, 11, 12, 13, 20, 21, 22, 23,
                          30, 31, 32, 33, 40, 41, 42, 43});
            write_floats(out, "eng.projective_information", {10, 20, 30, 40});
            ckpt::tar_finish(out);
        }
        check_capacity(root, 6, {0, 1, 2, 3});
        check_capacity(root, 2, {1, 3});
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        fs::remove_all(root);
        return 1;
    }
    fs::remove_all(root);
    std::puts("checkpoint_projective_adapt passed");
}
