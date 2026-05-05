# image-stitching

g++ src/*.cpp -Iinclude -o app
./app

nvc++ -std=c++17 -Iinclude -acc=gpu -o stitch src/*.cpp -lm
