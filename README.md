# image-stitching

g++ src/*.cpp -Iinclude -o app
./app

nvc++ -std=c++17 -Iinclude -Wall -Wextra -acc=gpu -gpu=managed -o stitch src/*.cpp -lm
