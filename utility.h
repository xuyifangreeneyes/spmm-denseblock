#include <vector>

template<typename T>
T* vec2ptr(const std::vector<T>& vec) {
    T* ptr = (T*) malloc(vec.size() * sizeof(T));
    for (int i = 0; i < vec.size(); ++i) {
        ptr[i] = vec[i];
    }
    return ptr;
}