#include "ConvolutionFilter.h"

ConvolutionFilter::ConvolutionFilter(int s)
{
	size = s;
	weights = std::vector<float>(size * size);
	factor = 1;
}

ConvolutionFilter::ConvolutionFilter(int s, std::vector<float>& w)
{
	size = s;
	weights = w;
	factor = calculateFactor();
}

void ConvolutionFilter::setWeights(const std::vector<float>& w)
{
	weights = w;
	factor = calculateFactor();
}

float ConvolutionFilter::calculateFactor()
{
	float weightSum = 0.0;
	for (float weight : weights) {
		weightSum += weight;
	}

	if (weightSum == 0.0f)
		weightSum = 1.0f;

	return 1.0 / weightSum;
}
