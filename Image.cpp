#include "Image.h"

Image::Image(const std::string& filename)
{
	// LETS USE QUICK CG TO LOAD THE DATA ARRAY
	unsigned long w = 0, h = 0;
	std::vector<QuickCG::ColorRGBA> image;
	QuickCG::loadImage(image, w, h, filename);

	width = w;
	height = h;
	data = std::vector<int>(width * height * 3);

	for (int x = 0; x < width * height; x++)
	{
		data[x * 3] = image[x].r;
		data[x * 3 + 1] = image[x].g;
		data[x * 3 + 2] = image[x].b;
	}
}

Image::Image(int w, int h)
{
    width = w;
    height = h;
    data = std::vector<int>(width * height * 3);

    for (int x = 0; x < width * height; x++)
    {
        data[x * 3] = 0;
        data[x * 3 + 1] = 0;
        data[x * 3 + 2] = 0;
    }
}

void Image::setData(const std::vector<int> d)
{
    data = d;
}

QuickCG::ColorRGBA Image::getPixel(int x, int y) const
{
	int r = data[x * 3 + y * width * 3];
	int g = data[x * 3 + y * width * 3 + 1];
	int b = data[x * 3 + y * width * 3 + 2];

	return QuickCG::ColorRGBA(r, g, b, 255);
}

void Image::setPixel(int x, int y, int r, int g, int b)
{
	data[x * 3 + y * width * 3] = r;
	data[x * 3 + y * width * 3 + 1] = g;
	data[x * 3 + y * width * 3 + 2] = b;
}

Image Image::applyConvolutionFilter(ConvolutionFilter& convolutionFilter, const bool onGPU) const
{
    Image newImage(width, height);
    int filterSize = convolutionFilter.getSize();

    if (!onGPU)
    {
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            {
                float red = 0.0, green = 0.0, blue = 0.0;

                //multiply every value of the filter with corresponding image pixel
                for (int filterY = 0; filterY < filterSize; filterY++)
                    for (int filterX = 0; filterX < filterSize; filterX++)
                    {
                        int imageX = (x - filterSize / 2 + filterX + width) % width;
                        int imageY = (y - filterSize / 2 + filterY + height) % height;

                        QuickCG::ColorRGBA origionalColor = getPixel(imageX, imageY);

                        float weight = convolutionFilter.getWeight(filterX, filterY);

                        red += origionalColor.r * weight;
                        green += origionalColor.g * weight;
                        blue += origionalColor.b * weight;
                    }

                //truncate values smaller than zero and larger than 255
                float factor = convolutionFilter.getFactor();
                float bias = 0.0f;

                red = std::min(std::max(int(factor * red + bias), 0), 255);
                green = std::min(std::max(int(factor * green + bias), 0), 255);
                blue = std::min(std::max(int(factor * blue + bias), 0), 255);

                newImage.setPixel(x, y, red, green, blue);
            }
    }
    else
    {
        try
        {
            cl::Buffer oldImageBuffer;
            cl::Buffer newImageBuffer;
            cl::Buffer convolutionWeights;

            // CREATE CONTEXT
            cl::Context context(CL_DEVICE_TYPE_DEFAULT);

            // LOAD PROGRAM
            cl::Program program(context, util::loadProgram("convolution.cl"), true);

            // GET QUEUE
            cl::CommandQueue queue(context);

            // CREATE FUNCTION
            auto convolution = cl::make_kernel<cl::Buffer, cl::Buffer, int, int, int, float, cl::Buffer>(program, "convolution");

            // COPY DATA FROM HOST TO DEVICE
            std::vector<int> imageData = getData();
            oldImageBuffer = cl::Buffer(context, begin(imageData), end(imageData), true);

            std::vector<float> filterWeights = convolutionFilter.getWeights();
            convolutionWeights = cl::Buffer(context, begin(filterWeights), end(filterWeights), true);

            // ALLOC MEMORY FOR RESULT
            newImageBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * width * height * 3);

            convolution(
                cl::EnqueueArgs(
                    queue,
                    cl::NDRange(width, height)),
                oldImageBuffer,
                newImageBuffer,
                width,
                height,
                convolutionFilter.getSize(),
                convolutionFilter.getFactor(),
                convolutionWeights
            );

            queue.finish();

            // COPY FROM DEVICE TO HOST
            std::vector<int> newImageData(width * height * 3);
            cl::copy(queue, newImageBuffer, begin(newImageData), end(newImageData));

            newImage.setData(newImageData);
        }
        catch (cl::Error err) {
            std::cout << "Exception\n";
            std::cerr
                << "ERROR: "
                << err.what()
                << "("
                << err.err()
                << ")"
                << std::endl;
        }
    }

	return newImage;
}
