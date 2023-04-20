#pragma once

#include <vector>

class ConvolutionFilter
{
private:
	int size;
	std::vector<float> weights;
	float factor;

public:
	ConvolutionFilter(int s);
	ConvolutionFilter(int s, std::vector<float>& w);

	float calculateFactor();

	std::vector<float> getWeights() { return weights; }
	void setWeights(const std::vector<float>& w);

	int getSize() const { return size; }
	float getWeight(int x, int y) const { return weights[x * size + y]; }

	float getFactor() const { return factor; };
};

