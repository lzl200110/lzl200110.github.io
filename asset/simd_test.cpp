#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <endian.h>
#include <execution>
#include <experimental/bits/simd.h>
#include <vector>
#include <experimental/simd>
#ifndef DEFINES_H
#define DEFINES_H
#include <iostream>
#include <chrono>
#define Tick   auto __begin = std::chrono::steady_clock::now();
#define ReTick __begin = std::chrono::steady_clock::now();
#define Tock                                                                                                                                                                       \
    {                                                                                                                                                                              \
        auto __end  = std::chrono::steady_clock::now();                                                                                                                            \
        auto __time = __end - __begin;                                                                                                                                             \
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(__time).count() << "ns" << std::endl;                                                                    \
    }

#endif
struct Pixel {
    explicit Pixel(uint32_t v) : a(v >> 24), r(v >> 16 & 0xFFu), g(v >> 8 & 0xFFu), b(v & 0xFFu) {}
    Pixel() = default;
    friend bool operator==(Pixel const& lsh, Pixel const& rsh) {
        return lsh.b == rsh.b && lsh.g == rsh.g && lsh.r == rsh.r && lsh.a == rsh.a;
    }
    uint8_t b, g, r, a;
};
using Image = std::vector<Pixel>;
void to_gray1(Image& img) {
    for (auto& pixel : img) {
        const auto gray = (pixel.r * 11 + pixel.g * 16 + pixel.b * 5) / 32;
        pixel.r = pixel.g = pixel.b = gray;
    }
}
void to_gray2(Image& img) {
    std::for_each(std::execution::unseq, img.begin(), img.end(), [](Pixel& pixel) {
        const auto gray = (pixel.r * 11 + pixel.g * 16 + pixel.b * 5) / 32;
        pixel.r = pixel.g = pixel.b = gray;
    });
}
namespace stdx = std::experimental;

using PixelSIMD   = stdx::fixed_size_simd<uint8_t, 4>;
using ImageSIMD   = std::vector<PixelSIMD>;
using Pixel32SIMD = stdx::fixed_size_simd<uint32_t, 4>;
using Pixel32Mask = PixelSIMD::mask_type;

const uint32_t gray_coeff[4]{5, 16, 11, 0};

void  to_gray3(ImageSIMD& img) {
    Pixel32Mask mask{};
    for (int i = 0; i < mask.size(); i++) {
        mask[i] = i < 3;
    }
    Pixel32SIMD gray_vector(gray_coeff, stdx::element_aligned);
    for (PixelSIMD& p8 : img) {
        const Pixel32SIMD pixel = p8;
        const uint32_t    gray  = stdx::reduce(pixel * gray_vector) / 32u;
        p8[0] = p8[1] = p8[2] = gray; // 这里应该运行一个simd双调选择(基于mask)，标准库目前没有，等待C++26吧
    }
}

using Pixel32 = uint32_t;
using Image32 = std::vector<uint32_t>;
using PixelV  = stdx::native_simd<Pixel32>;
void to_gray4(Image32& img) { 
    for (auto it = img.begin(); it < img.end(); it += PixelV::size()) {
        PixelV     p(&*it, stdx::element_aligned);
        const auto a     = p >> 24;
        const auto r     = p >> 16 & 0xFFu;
        const auto g     = p >> 8 & 0xFFu;
        const auto b     = p & 0xFFu;
        const auto grayv = (r * 11u + g * 16u + b * 5u) / 32u;
        p                = grayv | (grayv << 8) | (grayv << 16) | (a << 24);
        p.copy_to(&*it, stdx::element_aligned);
    }
}

int main() {
    Image img{};
    img.reserve(4096uz * 4096);
    //随机生成图片
    for (int i = 0; i < img.capacity(); i++) {
        img.emplace_back(std::rand());
    }
    auto      img2 = img;
    ImageSIMD img3{};
    img3.reserve(img.size());
    for (const auto& i : img) {
        stdx::fixed_size_simd<uint8_t, 4> tmp{reinterpret_cast<const uint8_t*>(&i), stdx::element_aligned};
        img3.push_back(std::move(tmp));
    }
    Image32 img4{};
    img4.reserve(img.size());
    for (auto const& i : img) {
        img4.push_back(*reinterpret_cast<const uint32_t*>(&i)); // 小端序的问题，这里是0XAARRGGBB,
    }
    Tick;
    to_gray1(img);
    Tock;
    ReTick;
    to_gray2(img2);
    Tock;
    // 检查 to_gray2 实现是否正确
    // for (int i = 0; i < img.size(); i++) {
    //     if (img[i] != img2[i]) {
    //         std::cerr << "dismatch on img2, index is " << i << std::endl;
    //         return -1;
    //     }
    // }
    ReTick;
    to_gray3(img3);
    Tock;
    // 检查 to_gray3 实现是否正确
    // for (int i = 0; i < img.size(); i++) {
    //     stdx::fixed_size_simd<uint8_t, 4> tmp{reinterpret_cast<const uint8_t*>(&img[i]), stdx::element_aligned};
    //     if (!stdx::all_of(tmp == img3[i])) {
    //         std::cerr << "dismatch on img3, index is " << i << std::endl;
    //         return -1;
    //     }
    // }
    ReTick;
    to_gray4(img4);
    Tock;
    // 检查 to_gray4 实现是否正确
    // for (int i = 0; i < img.size(); i++) {
    //     Pixel p{img4[i]};
    //     if (img[i] != p) {
    //         std::cerr << "dismatch on img4, index is " << i << std::endl;
    //         return -1;
    //     }
    // }
}