
__kernel void convolution(__global const int* inputImage,
                           __global int *outputImage,
                           const int width,
                           const int height,
                           const int kernelSize,
                           const float factor,
                           __global const float* kernelWeights) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    int id = x * 3 + y * width * 3;

    float red = 0.0, green = 0.0, blue = 0.0;

    for (int filterY = 0; filterY < kernelSize; filterY++)
    {
        for (int filterX = 0; filterX < kernelSize; filterX++)
        {
            int imageX = (x - kernelSize / 2 + filterX + width) % width;
            int imageY = (y - kernelSize / 2 + filterY + height) % height;

            float weight = kernelWeights[filterX * kernelSize + filterY];

            int currentId = imageX * 3 + imageY * width * 3;

            red += inputImage[currentId] * weight;
            green += inputImage[currentId + 1] * weight;
            blue += inputImage[currentId + 2] * weight;
        }
    }

    red = factor * red;

    if(red < 0) red = 0;
    else if(red > 255) red = 255;

    green = factor * green;

    if(green < 0) green = 0;
    else if(green > 255) green = 255;

    blue = factor * blue;

    if(blue < 0) blue = 0;
    else if(blue > 255) blue = 255;

    outputImage[id] = red;
    outputImage[id + 1] = green;
    outputImage[id + 2] = blue;
}