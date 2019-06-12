clang++ -O3 -std=c++17 -stdlib=libc++ -Ithird_party/x64-osx/include main.cpp
./a.out
pnmtopng pixels.ppm > pixels.png
