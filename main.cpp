#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <ios>
#include <iostream>
#include <numeric>
#include <sstream>
#include <tuple>
#include <vector>

using Color = std::array<uint8_t, 3>;
using Point = std::array<float, 3>;
using Ray = std::array<float, 3>;

struct Material {
  Color color;
};

const std::vector<Material> kMaterials = {
    Material{{255, 0, 0}},
    Material{{0, 255, 0}},
    Material{{0, 0, 255}},
};

struct Camera {
  Point position;
  Ray direction;
  float fov;
};

class Pixels {
 public:
  Pixels(int w, int h) : shape_{w, h}, colors_(w * h) {}

  Color& get(int x, int y) {
    auto [w, h] = shape_;
    return colors_.at(x + w * y);
  }

  void set(int x, int y, Color c) {
    auto [w, h] = shape_;
    colors_.at(x + w * y) = std::move(c);
  }

 private:
  std::array<int, 2> shape_;
  std::vector<Color> colors_;
};

template <typename Value>
class Voxels {
 public:
  Voxels(int w, int h, int d) : shape_{w, h, d}, values_(w * h * d) {}

  Value get(int x, int y, int z) {
    auto [w, h, d] = shape_;
    return values_.at(x + w * (y + h * z));
  }

  void set(int x, int y, int z, Value v) {
    auto [w, h, d] = shape_;
    values_.at(x + w * (y + h * z)) = std::move(v);
  }

 private:
  std::array<int, 3> shape_;
  std::vector<Value> values_;
};

template <typename Fn>
void march(const Point& from, const Ray& direction, Fn&& fn) {
  auto [x, y, z] = from;
  auto [u, v, w] = direction;

  // The signs of the ray direction vector components.
  auto sx = std::signbit(u);
  auto sy = std::signbit(v);
  auto sz = std::signbit(w);

  // The ray distance traveled per unit in each direction.
  auto norm = std::sqrt(u * u + v * v + w * w);
  auto dx = norm / std::abs(u);
  auto dy = norm / std::abs(v);
  auto dz = norm / std::abs(w);

  // The ray distance to the next intersection in each direction.
  auto lx = (sx ? (x - std::floor(x)) : (1 + std::floor(x) - x)) * dx;
  auto ly = (sy ? (y - std::floor(y)) : (1 + std::floor(y) - y)) * dy;
  auto lz = (sz ? (z - std::floor(z)) : (1 + std::floor(z) - z)) * dz;

  // Advance voxel indices that intersect with the given ray.
  float distance = 0.0;
  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);
  int iz = static_cast<int>(z);
  for (;;) {
    if (!fn(ix, iy, iz, distance)) {
      break;
    }

    // Advance one voxel in the direction of nearest intersection.
    if (lx <= ly && lx <= lz) {
      distance = lx;
      ix += sx ? -1 : 1;
      lx += dx;
    } else if (ly <= lz) {
      distance = ly;
      iy += sy ? -1 : 1;
      ly += dy;
    } else {
      distance = lz;
      iz += sz ? -1 : 1;
      lz += dz;
    }
  }
}

class Timer {
 public:
  Timer(std::string msg)
      : msg_(std::move(msg)),
        start_(std::chrono::high_resolution_clock::now()) {}
  ~Timer() {
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = finish - start_;
    std::cout << msg_ << " took: " << duration.count() << "s\n";
  }

 private:
  std::string msg_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

template <typename Iter, typename Fn>
void parallel_for(Iter begin, Iter end, Fn&& fn) {
  if (begin != end) {
    auto future = std::async(std::launch::async, fn, *begin);
    parallel_for(++begin, end, fn);
    future.wait();
  }
}

int main() {
  // Create the voxels data structure.
  constexpr auto kVoxelsSize = 128;
  Voxels<uint32_t> voxels(kVoxelsSize, kVoxelsSize, kVoxelsSize);

  // Initialize all border voxels to some material.
  for (int i = 0; i < kVoxelsSize; i += 1) {
    for (int j = 0; j < kVoxelsSize; j += 1) {
      for (int k = 0; k < kVoxelsSize; k += 1) {
        if (i == 0 || i == kVoxelsSize - 1) {
          voxels.set(i, j, k, 1);
        } else if (j == 0 || j == kVoxelsSize - 1) {
          voxels.set(i, j, k, 2);
        } else if (k == 0 || k == kVoxelsSize - 1) {
          voxels.set(i, j, k, 3);
        }
      }
    }
  }

  // Create a camera.
  Camera camera{{60.0, 64.0, 64.0}, {1.0, 0.0, 0.0}, 1.570796};

  // Create the pixels data structure.
  constexpr auto kPixelsSize = 1024;
  Pixels pixels(kPixelsSize, kPixelsSize);

  // Cast a ray through each pixel to identify its color.
  std::vector<Ray> rays;
  {
    Timer timer("prepare_rays");
    for (int i = 0; i < kPixelsSize; i += 1) {
      for (int j = 0; j < kPixelsSize; j += 1) {
        auto x = (i + 0.5f) / kPixelsSize;
        auto y = (j + 0.5f) / kPixelsSize;
        auto t = camera.fov * (x - 0.5f);
        auto p = camera.fov * (y - 0.5f);
        rays.emplace_back(Ray{
            std::cos(t) * std::cos(p),
            std::sin(t) * std::cos(p),
            std::sin(p),
        });
      }
    }
  }

  {
    Timer timer("trace_rays");
    std::vector<int> j_range(kPixelsSize);
    std::iota(j_range.begin(), j_range.end(), 0);
    parallel_for(j_range.begin(), j_range.end(), [&](int j) {
      for (int i = 0; i < kPixelsSize; i += 1) {
        march(
            camera.position,
            rays.at(i + kPixelsSize * j),
            [&](int ix, int iy, int iz, float distance) {
              if (ix < 0 || iy < 0 || iz < 0) {
                return false;
              }
              if (ix >= kVoxelsSize || iy >= kVoxelsSize || iz >= kVoxelsSize) {
                return false;
              }
              if (auto mat = voxels.get(ix, iy, iz)) {
                pixels.set(i, j, kMaterials.at(mat - 1).color);
                return false;
              }
              return true;
            });
      }
    });
  }

  // Dump the pixels to a PNG file.
  std::cout << "Writing pixels to output image." << std::endl;
  std::vector<uint8_t> data(3 * kPixelsSize * kPixelsSize);
  for (int i = 0; i < kPixelsSize; i += 1) {
    for (int j = 0; j < kPixelsSize; j += 1) {
      auto color = pixels.get(i, j);
      auto offset = i + kPixelsSize * j;
      data.at(3 * offset) = color.at(0);
      data.at(3 * offset + 1) = color.at(1);
      data.at(3 * offset + 2) = color.at(2);
    }
  }

  // Make the header.
  std::stringstream ss;
  ss << "P6\n";
  ss << kPixelsSize << " " << kPixelsSize << "\n";
  ss << "255\n";

  // Dump the data.
  std::fstream img("pixels.ppm", std::ios_base::out | std::ios_base::binary);
  img.write(ss.str().c_str(), ss.str().length());
  img.write(reinterpret_cast<char*>(&data[0]), data.size());
  img.close();
  std::cout << "All done." << std::endl;
}