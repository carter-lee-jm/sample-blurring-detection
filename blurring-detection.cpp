#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <conio.h>
struct Result {
	std::string filename;
	double threshold;
	double max;
	double min;
	double mean;
	double stddev;
	double sum;
};
Result check(std::string filename, float thold)
{
	// 讀取影像
	//std::string filename = R"(C:\code\sample-polarization-image\out\build\x64-Clang-Release-2019\RelWithDebInfo\0\101_1547.png)";
	cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	cv::Rect roi(1124, 802, 150, 150);
	cv::Mat roiImage = image(roi);
	image = roiImage;
	if (image.empty())
	{
		std::cout << "Could not open or find the image" << std::endl;
		Result err;
		err.max = -1;
		return err;
	}

	cv::Mat padded;
	int m = cv::getOptimalDFTSize(image.rows);
	int n = cv::getOptimalDFTSize(image.cols);
	cv::copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));


	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexImage;
	cv::merge(planes, 2, complexImage);

	// 進行傅立葉變換
	cv::dft(complexImage, complexImage);

	// 分解實數部分和虛數部分
	cv::split(complexImage, planes);

	// 計算幅度並轉換到對數尺度
	cv::Mat magnitude;
	cv::magnitude(planes[0], planes[1], magnitude);
	magnitude += cv::Scalar::all(1);
	cv::log(magnitude, magnitude);

	cv::Mat fordisplay;
	cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);
	cv::imshow("fordisplay", magnitude);


	// 重新調整四個角落
	int cx = magnitude.cols / 2;
	int cy = magnitude.rows / 2;
	cv::Mat tmp;
	cv::Mat q0(magnitude, cv::Rect(0, 0, cx, cy));   // Top-Left
	cv::Mat q1(magnitude, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magnitude, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magnitude, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
	q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
	q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

	// 正規化到可顯示範圍
	cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

	// 顯示原始影像和頻譜
	cv::imshow("Input Image", image);
	cv::imshow("Spectrum magnitude", magnitude);

	// Step 2: Set the spectral cut-off threshold
	float threshold = 10;  // This is your cut-off threshold, you can adjust it based on your needs
	threshold = thold;
	cv::Mat mask = magnitude > threshold;
	mask.convertTo(mask, CV_32F);
	// Apply the mask to the complex image
	planes[0] = planes[0].mul(mask);
	planes[1] = planes[1].mul(mask);

	// Step 3: Perform the inverse DFT
	cv::Mat inverseTransform;
	cv::merge(planes, 2, inverseTransform);
	cv::idft(inverseTransform, inverseTransform);

	// Split real and imaginary planes
	cv::split(inverseTransform, planes);  // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

	// Compute magnitude of inverse DFT
	cv::magnitude(planes[0], planes[1], inverseTransform);  // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)

	// Normalize the image
	cv::normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);

	// Display the result
	cv::imshow("Reconstructed Image", inverseTransform);


	// Compute the absolute difference between the original image and the reconstructed image
	cv::Mat originalFloat;
	image.convertTo(originalFloat, CV_32F);
	cv::normalize(originalFloat, originalFloat, 0, 1, cv::NORM_MINMAX);


	//std::cout << "originalFloat size: " << originalFloat.size() << std::endl;
	//std::cout << "inverseTransform size: " << inverseTransform.size() << std::endl;
	//std::cout << "originalFloat type: " << originalFloat.type() << std::endl;
	//std::cout << "inverseTransform type: " << inverseTransform.type() << std::endl;
	//std::cout << "originalFloat channels: " << originalFloat.channels() << std::endl;
	//std::cout << "inverseTransform channels: " << inverseTransform.channels() << std::endl;


	inverseTransform = inverseTransform(cv::Rect(0, 0, image.cols, image.rows));
	cv::Mat diff = cv::abs(originalFloat - inverseTransform);

	// Display the difference
	cv::imshow("Difference", diff);

	// Optionally, compute some statistics about the difference
	double minVal, maxVal;
	cv::minMaxLoc(diff, &minVal, &maxVal);
	//std::cout << "input filename = " << filename << std::endl;
	//std::cout << "threshold:" << threshold << std::endl;
	//std::cout << "Min difference: " << minVal << std::endl;
	//std::cout << "Max difference: " << maxVal << std::endl;

	// Compute the mean difference
	cv::Scalar meanDiff = cv::mean(diff);
	//std::cout << "Mean difference: " << meanDiff.val[0] << std::endl;

	// Compute the standard deviation of the difference
	cv::Scalar mean, stddev;
	cv::meanStdDev(diff, mean, stddev);
	//std::cout << "Standard deviation of difference: " << stddev.val[0] << std::endl;

	// Compute the sum of the difference
	cv::Scalar sumDiff = cv::sum(diff);
	//std::cout << "Sum of difference: " << sumDiff.val[0] << std::endl;


	cv::waitKey(1);

	Result ret;
	ret.max = maxVal;
	ret.min = minVal;
	ret.mean = meanDiff.val[0];
	ret.stddev = stddev.val[0];
	ret.sum = sumDiff.val[0];
	return ret;
}
int writevecttofile(std::vector<Result> results,std::string filename)
{
	std::ofstream outputFile(filename, std::ofstream::trunc);
	if (!outputFile) {
		std::cerr << "Failed to open output file." << std::endl;
		return 1;
	}

	// Write the title row to the CSV file
	outputFile << "Threshold,Mean,Standard Deviation,Sum" << std::endl;

	for (const auto& ret : results) {
		outputFile << ret.threshold << "," << ret.mean << "," << ret.stddev << "," << ret.sum << std::endl;
	}

	outputFile.close();
	std::cout << "Results written to " << filename << std::endl;
	return 0;

}


int infileleveltocsv(std::vector<Result> results, std::string filename, double threshold)
{
	std::ofstream outputFile(filename, std::ofstream::trunc);
	if (!outputFile) {
		std::cerr << "Failed to open output file." << std::endl;
		return 1;
	}

	// Write the title row to the CSV file
	outputFile << "Filename,Threshold,Mean,Standard Deviation,Sum" << std::endl;

	for (const auto& ret : results) {
		outputFile <<ret.filename<< "," << ret.threshold << "," << ret.mean << "," << ret.stddev << "," << ret.sum << std::endl;
	}

	outputFile.close();
	std::cout << "Results written to " << filename << std::endl;
	return 0;

}
std::string extractNumbers(std::string str) {
	std::string number;
	std::size_t underscorePos = str.find('_');
	if (underscorePos != std::string::npos) {
		number = str.substr(0, underscorePos);
	}
	return number;
}

std::string getFilenameWithoutExtension(const std::string& filePath) {
	std::filesystem::path path(filePath);
	return extractNumbers(path.stem().string());
}

std::string padNumberWithZeros(const std::string& number, int width) {
	std::stringstream ss;
	ss << std::setw(width) << std::setfill('0') << number;
	return ss.str();
}
bool compareResultsByFilename(const Result& a, const Result& b) {
	return a.filename < b.filename;
}
void draw(std::vector<Result> results)
{
	// Sort the results by filename
	std::sort(results.begin(), results.end(), compareResultsByFilename);

	// Prepare data for plotting
	std::vector<double> xValues;
	std::vector<double> yValues;
	for (const auto& result : results) {
		xValues.push_back(std::stod(result.filename));
		yValues.push_back(result.mean);
	}

	// Create a line chart using OpenCV
	int chartWidth = 1800;
	int chartHeight = 768;
	cv::Mat chart(chartHeight, chartWidth, CV_8UC3, cv::Scalar(255, 255, 255));

	// Find the minimum and maximum values for scaling the chart
	double minX = *std::min_element(xValues.begin(), xValues.end());
	double maxX = *std::max_element(xValues.begin(), xValues.end());
	double minY = *std::min_element(yValues.begin(), yValues.end());
	double maxY = *std::max_element(yValues.begin(), yValues.end());

	// Draw the chart axes
	cv::line(chart, cv::Point(50, chartHeight - 50), cv::Point(50, 50), cv::Scalar(0, 0, 0), 2);
	cv::line(chart, cv::Point(50, chartHeight - 50), cv::Point(chartWidth - 50, chartHeight - 50), cv::Scalar(0, 0, 0), 2);

	// Draw the data points
	for (size_t i = 0; i < xValues.size(); i++) {
		int x = static_cast<int>((xValues[i] - minX) / (maxX - minX) * (chartWidth - 100) + 50);
		int y = static_cast<int>(chartHeight - 50 - (yValues[i] - minY) / (maxY - minY) * (chartHeight - 100));
		cv::circle(chart, cv::Point(x, y), 5, cv::Scalar(255, 0, 0), -1);
	}

	// Display the chart
	cv::imshow("Line Chart", chart);
	cv::waitKey(0);
}

void drawbl(std::vector<Result> results)
{
	// Sort the results by filename
	std::sort(results.begin(), results.end(), compareResultsByFilename);

	// Prepare data for plotting
	std::vector<double> xValues;
	std::vector<double> yValues;
	for (const auto& result : results) {
		xValues.push_back(std::stod(result.filename));
		yValues.push_back(result.threshold);
	}

	// Create a line chart using OpenCV
	int chartWidth = 1800;
	int chartHeight = 768;
	cv::Mat chart(chartHeight, chartWidth, CV_8UC3, cv::Scalar(255, 255, 255));

	// Find the minimum and maximum values for scaling the chart
	double minX = *std::min_element(xValues.begin(), xValues.end());
	double maxX = *std::max_element(xValues.begin(), xValues.end());
	double minY = *std::min_element(yValues.begin(), yValues.end());
	double maxY = *std::max_element(yValues.begin(), yValues.end());

	// Draw the chart axes
	cv::line(chart, cv::Point(50, chartHeight - 50), cv::Point(50, 50), cv::Scalar(0, 0, 0), 2);
	cv::line(chart, cv::Point(50, chartHeight - 50), cv::Point(chartWidth - 50, chartHeight - 50), cv::Scalar(0, 0, 0), 2);

	// Draw the data points
	for (size_t i = 0; i < xValues.size(); i++) {
		int x = static_cast<int>((xValues[i] - minX) / (maxX - minX) * (chartWidth - 100) + 50);
		int y = static_cast<int>(chartHeight - 50 - (yValues[i] - minY) / (maxY - minY) * (chartHeight - 100));
		cv::circle(chart, cv::Point(x, y), 5, cv::Scalar(255, 0, 0), -1);
	}

	// Display the chart
	cv::imshow("Line Chart", chart);
	cv::waitKey(0);
}
void shiftDFT(cv::Mat& magnitude, cv::Mat& shifted)
{
	// Rearrange the quadrants of Fourier image so that the origin is at the image center
	int cx = magnitude.cols / 2;
	int cy = magnitude.rows / 2;

	cv::Mat q0(magnitude, cv::Rect(0, 0, cx, cy));
	cv::Mat q1(magnitude, cv::Rect(cx, 0, cx, cy));
	cv::Mat q2(magnitude, cv::Rect(0, cy, cx, cy));
	cv::Mat q3(magnitude, cv::Rect(cx, cy, cx, cy));

	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	shifted = magnitude.clone();
}
//this result is very bad
float BlurredLevel_ng(cv::Mat img, double thresh) {
	cv::Mat img_gray, img_float, magnitude, shifted;
	img_gray = img;
	img_gray.convertTo(img_float, CV_32F);
	cv::Mat complexImg;
	dft(img_float, complexImg, cv::DFT_COMPLEX_OUTPUT);
	cv::Mat planes[2];
	cv::split(complexImg, planes);
	cv::Mat mag;
	cv::magnitude(planes[0], planes[1], mag);
	mag += cv::Scalar::all(1);
	log(mag, mag);
	shiftDFT(mag, shifted);
	cv::Mat fordisplay;
	cv::normalize(shifted, fordisplay, 0, 1, cv::NORM_MINMAX);
	cv::imshow("shifted", fordisplay);
	//cv::waitKey();
	int rows = img.rows;
	int cols = img.cols;
	int size = cv::min(rows, cols) * 0.05;
	int crow = rows / 2;
	int ccol = cols / 2;
	cv::Rect roi = cv::Rect(ccol - size / 2, crow - size / 2, size, size);
	shifted(roi).setTo(0);
	cv::Mat inverseTransform;
	idft(shifted, inverseTransform, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	cv::Scalar mean = cv::mean(inverseTransform);
	return mean[0];
}



void drawThreshold(cv::Mat& image, float threshold) {
	// 設定文字參數
	double fontScale = image.cols * 0.002; // 字型大小約為畫面的10%
	int baseline = 0;
	int thickness = 2;
	int padding = static_cast<int>(fontScale * 2); // 文字與上方的間距

	// 將 threshold 轉換成字串
	std::stringstream stream;
	stream << std::fixed << std::setprecision(9) << threshold;
	std::string text = "Threshold: " + stream.str();

	// 取得文字大小
	cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);

	// 計算文字位置
	int x = image.cols * 0.05; // 文字位於畫面左邊 5% 的位置
	int y = padding + textSize.height; // 文字位於上方 10% 加上 padding 的位置

	// 繪製文字
	cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(100), thickness);
}

//this result is good
double calculateBlur(cv::Mat image) {
	cv::Mat gray;
	gray = image;

	cv::Mat laplacian;
	cv::Laplacian(gray, laplacian, CV_64F);

	cv::Scalar mu, sigma;
	cv::meanStdDev(laplacian, mu, sigma);

	double focusMeasure = sigma.val[0] * sigma.val[0];
	return focusMeasure;
}

int main(int argc, char** argv) {
	std::string filestr = "49_1723";
	std::string foldername = R"(C:\code\sample-blurring-detection\image\0\)";
	std::string filename = foldername + filestr + ".png";

	std::vector<std::string> pngFiles;
	for (const auto& entry : std::filesystem::directory_iterator(foldername)) {
		if (entry.is_regular_file() && entry.path().extension() == ".png") {
			pngFiles.push_back(entry.path().string());
		}
	}


	double max = 0;
	std::string goodfile;
	//for (int i = 4; i < 11; i++)
	//{
	//	std::vector<Result> results;
	//	float threshold = 0.1*i;
	//	for (const auto& pngFile : pngFiles) {

	//		Result ret = check(pngFile, threshold);
	//		ret.threshold = threshold;
	//		ret.filename = padNumberWithZeros(getFilenameWithoutExtension(pngFile), 3);
	//		if (ret.mean > max)
	//		{
	//			max = ret.mean;
	//			goodfile = pngFile;
	//		}
	//		results.push_back(ret);
	//		std::cout << threshold << "\t" << ret.mean << "\t" << ret.stddev << "\t" << ret.sum << std::endl;
	//	}
	//	infileleveltocsv(results, std::to_string(i)+".csv", 0.2);
	//	cv::Mat goodimg = cv::imread(goodfile);
	//	cv::imshow("focus", goodimg);
	//	std::cout << goodfile << std::endl;
	//	cv::waitKey(1);
	//	draw(results);
	//}



	std::vector<Result> results;
	for (const auto& pngFile : pngFiles)
	{
		cv::Mat image = cv::imread(pngFile, cv::IMREAD_GRAYSCALE);
		cv::Rect roi(1021, 578, 100, 100);
		cv::Mat roiImage = image(roi);
		image = roiImage;
		float threshold= calculateBlur(image);
		Result ret;
		ret.filename= padNumberWithZeros(getFilenameWithoutExtension(pngFile), 3);
		ret.threshold = threshold;
		drawThreshold(image, threshold);
		cv::imshow("Image with Threshold", image);
		cv::waitKey(1);
		if (ret.threshold > max)
		{
			max = ret.threshold;
			goodfile = pngFile;
		}
		results.push_back(ret);
	}
	std::cout << goodfile << std::endl;
	cv::Mat goodimg = cv::imread(goodfile);
	cv::namedWindow("focus", cv::WINDOW_NORMAL);
	cv::imshow("focus", goodimg);
	cv::waitKey();
	//std::string file1 = R"(C:\code\sample-polarization-image\out\build\x64-Clang-Release-2019\RelWithDebInfo\2023_05_18_18_52_41_1759.jpg)";
	//std::string file2 = R"(C:\code\sample-polarization-image\out\build\x64-Clang-Release-2019\RelWithDebInfo\2023_05_18_18_52_41_1759.jpg)";
	//cv::Mat image = cv::imread(file1, cv::IMREAD_GRAYSCALE);
	//cv::Rect roi(1364, 766, 200, 200);
	//cv::Mat roiImage = image(roi);
	//image = roiImage;
	//float threshold = calculateBlur(image);
	//std::cout << threshold << std::endl;

	//image = cv::imread(file2, cv::IMREAD_GRAYSCALE);
	//roiImage = image(roi);
	//image = roiImage;
	//threshold = calculateBlur(image);
	//std::cout << threshold << std::endl;

	infileleveltocsv(results, "bl.csv", 0.2);
	drawbl(results);
	return 0;
}