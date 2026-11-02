#include "reference.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
int main() {
    float a[] = {1, -2, 3.5f, 0};
    float b[] = {2, 4, -1.5f, 9};
    float output[4] = {};
    vector_add_cpu(a, b, output, 4);
    const float expected[] = {3, 2, 2, 9};
    for (int i = 0; i < 4; ++i) assert(close_enough(output[i], expected[i]));
    bool thrown = false;
    try { vector_add_cpu(nullptr, b, output, 4); }
    catch (const std::invalid_argument&) { thrown = true; }
    assert(thrown);
    std::cout << "PASS: CPU reference correctness and error path\n";
}
