#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
void endian() {
    union U {
        short v;
        char  c;
    } u;
    u.v = 0x0102;
    if (u.c == 0x02) {
        std::cout << "little endian";
    } else {
        std::cout << "big endian";
    }
}

template <typename T>
T brute_byteswap(T value) noexcept {
    auto value_representation = std::bit_cast<std::array<std::byte, sizeof(T)>>(value); //将value 转化为byte数组
    std::ranges::reverse(value_representation);                                         // reverse
    return std::bit_cast<T>(value_representation);
}

constexpr inline bool is_system_little_endian = (std::endian::little == std::endian::native);

template <typename T>
T byteswap(T value) {
    return value;
}

template <typename T>
    requires(sizeof(T) > 1 && is_system_little_endian)
T byteswap(T value) {
    return brute_byteswap(value);
}

struct Person {
    int    a;
    double b;
    char   c;
};

template <typename T>
std::byte* find_next_align(const std::byte* ptr) {
    auto u_ptr = reinterpret_cast<uint64_t>(ptr);
    u_ptr      = (u_ptr + ((1 << sizeof(T)) - 1)) & (~((1 << sizeof(T)) - 1));
    return reinterpret_cast<std::byte*>(u_ptr);
}
std::vector<std::byte> serialize(const Person& p) {
    std::vector<std::byte> res(sizeof(p));
    auto                   ptr = res.data();
    auto [a, b, c]             = p;
    a                          = byteswap(a);
    b                          = byteswap(b);
    c                          = byteswap(c);

    memcpy(ptr, &a, sizeof(a));

    ptr += sizeof(a);
    ptr = find_next_align<decltype(b)>(ptr);
    memcpy(ptr, &b, sizeof(b));

    ptr += sizeof(b);
    ptr = find_next_align<decltype(c)>(ptr);
    memcpy(ptr, &c, sizeof(c));

    return res;
};

Person deserialize(const std::vector<std::byte>& seq) {
    Person           p;
    const std::byte* ptr = seq.data();
    p.a                  = byteswap(*(int*)ptr);

    ptr += sizeof(p.a);
    ptr = find_next_align<decltype(p.b)>(ptr);
    p.b = byteswap(*(double*)ptr);

    ptr += sizeof(p.b);
    ptr = find_next_align<decltype(p.c)>(ptr);
    p.c = byteswap(*(char*)ptr);

    return p;
};
void print(const Person& p) {
    std::cout << p.a << '\t' << p.b << '\t' << p.c << '\n';
}
int main() {
    Person p{.a = 1, .b = 2.3, .c = 'f'};
    auto   bytes = serialize(p);
    auto   p2    = deserialize(bytes);
    print(p);
    print(p2);
}
