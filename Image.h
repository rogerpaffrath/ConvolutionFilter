#pragma once

#define __CL_ENABLE_EXCEPTIONS
#define NOMINMAX

#include "CL/cl.hpp"

#include "util.hpp"
#include "quickcg.h"

#include "ConvolutionFilter.h"

#include <cmath>
#include <string>

class Image
{
private:
    int width, height;
    std::vector<int> data;

public:
    Image(const std::string& filename);
    Image(int w, int h);

    int getWidth() { return width; }
    int getHeight() { return height; }

    std::vector<int>getData() const { return data; }
    void setData(const std::vector<int> d);

    QuickCG::ColorRGBA getPixel(int x, int y) const;
    void setPixel(int x, int y, int r, int g, int b);

    Image applyConvolutionFilter(ConvolutionFilter& convolutionFilter, const bool onGPU = false) const;
};

