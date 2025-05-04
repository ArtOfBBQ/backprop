APP_NAME="nnet"

FILENAMES="
src/main.c
src/toknnet.c
"

COMPILER_ARGS="
-D NDEBUG
-Wall
-O2
-x c
-std=c11
"

rm -r build/*

if gcc $COMPILER_ARGS $FILENAMES -o build/$APP_NAME;
then
    echo "compilation successful"
    (cd build && ./$APP_NAME)
else
    echo "failed to compile!"
fi

