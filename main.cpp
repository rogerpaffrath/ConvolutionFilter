#include "Image.h"

#undef main

using namespace std;

int main()
{
	cout << "Where would you like to run the filter? \n";
	cout << "1 - HOST \n";
	cout << "2 - DEVICE \n";
	cout << "\n";
	cout << "Type 1 or 2: ";

	string answer;

	getline(cin, answer);

	while (answer != "1" && answer != "2")
	{
		cout << "\n";
		cout << "Invalid input. Please, type 1 or 2: ";
		getline(cin, answer);
	}

	Image originalImage("pics/rogerinho.png");

	int w = originalImage.getWidth();
	int h = originalImage.getHeight();

	QuickCG::screen(w, h, 0, "Filters (convolution)");

	ConvolutionFilter convolutionFilter(3);
	convolutionFilter.setWeights({ 1,0,-1,2,0,-2,1,0,-1 });

	util::Timer timer;

	Image newImage = originalImage.applyConvolutionFilter(convolutionFilter, (answer == "2"));

	double time = static_cast<double>(timer.getTimeMilliseconds()) / 1.0;

	cout << "\n";
	printf("For a %d x %d image the %s took %lf miliseconds.\n", w, h, (answer == "1") ? "HOST" : "DEVICE", time);

	// draw on screen
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			QuickCG::ColorRGBA rgb = newImage.getPixel(x, y);
			QuickCG::pset(x, y, rgb);
		}

	//redraw & sleep
	QuickCG::redraw();
	QuickCG::sleep();
}