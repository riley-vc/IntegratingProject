#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <iomanip>

using namespace cv;
using namespace std;
using namespace chrono;

Mat kuwaharaFilter(const Mat& src, int kernelSize) {
    Mat result = src.clone();
    int radius = kernelSize / 2;
    int subRegionSize = radius + 1;

    // Process each pixel
    for (int y = radius; y < src.rows - radius; y++) {
        for (int x = radius; x < src.cols - radius; x++) {
            double minVariance = numeric_limits<double>::max();
            Vec3d bestMean(0, 0, 0);

            // upper left, lower left, lower right, upper right
            int offsetsX[] = { -radius, 0, -radius, 0 };
            int offsetsY[] = { -radius, -radius, 0, 0 };

            // Check each quadrant
            for (int q = 0; q < 4; q++) {
                int startX = x + offsetsX[q];
                int startY = y + offsetsY[q];

                Vec3d mean(0, 0, 0);
                Vec3d variance(0, 0, 0);
                int count = 0;

                // Calculate mean for quadrant
                for (int dy = 0; dy < subRegionSize; dy++) {
                    for (int dx = 0; dx < subRegionSize; dx++) {
                        int px = startX + dx;
                        int py = startY + dy;

                        if (px >= 0 && px < src.cols && py >= 0 && py < src.rows) {
                            Vec3b pixel = src.at<Vec3b>(py, px);
                            mean[0] += pixel[0];
                            mean[1] += pixel[1];
                            mean[2] += pixel[2];
                            count++;
                        }
                    }
                }

                mean /= count;

                // Calculate variance for quadrant
                for (int dy = 0; dy < subRegionSize; dy++) {
                    for (int dx = 0; dx < subRegionSize; dx++) {
                        int px = startX + dx;
                        int py = startY + dy;

                        if (px >= 0 && px < src.cols && py >= 0 && py < src.rows) {
                            Vec3b pixel = src.at<Vec3b>(py, px);
                            variance[0] += pow(pixel[0] - mean[0], 2);
                            variance[1] += pow(pixel[1] - mean[1], 2);
                            variance[2] += pow(pixel[2] - mean[2], 2);
                        }
                    }
                }

                variance /= count;
                double totalVariance = variance[0] + variance[1] + variance[2];

                // Keep quadrant with minimum variance
                if (totalVariance < minVariance) {
                    minVariance = totalVariance;
                    bestMean = mean;
                }
            }

            // Set output pixel to the mean with minimum variance
            result.at<Vec3b>(y, x) = Vec3b(
                saturate_cast<uchar>(bestMean[0]),
                saturate_cast<uchar>(bestMean[1]),
                saturate_cast<uchar>(bestMean[2])
            );
        }
    }

    return result;
}

int main() {

    string inputPath = "test.jpg";
    string outputPath = "output.jpg";  

    // Change kernel size variable here
    int kernelSize = 7;


    Mat image = imread(inputPath, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Error: Could not read image from " << inputPath << endl;
        return -1;
    }

    cout << "Original image size: " << image.cols << " x " << image.rows << endl;

    // Resize image for faster processing commented out for now 
    /*
    int maxDimension = 800;

    if (maxDimension > 0 && (image.cols > maxDimension || image.rows > maxDimension)) {
        double scale;
        if (image.cols > image.rows) {
            scale = (double)maxDimension / image.cols;
        }
        else {
            scale = (double)maxDimension / image.rows;
        }

        int newWidth = (int)(image.cols * scale);
        int newHeight = (int)(image.rows * scale);

        resize(image, image, Size(newWidth, newHeight), 0, 0, INTER_LINEAR);
        cout << "Resized to: " << image.cols << " x " << image.rows << endl;
    }
    */
    cout << "Processing image with Kuwahara filter (kernel size: " << kernelSize << ")..." << endl;


    // Start timing
    auto startTime = high_resolution_clock::now();

    Mat filtered = kuwaharaFilter(image, kernelSize);

    // End timing
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(endTime - startTime);

    cout << "Kuwahara filter completed in " << duration.count() << " ms ";
    cout << "(" << fixed << setprecision(2) << duration.count() / 1000.0 << " seconds)" << endl;

    // Create side-by-side comparison
    Mat comparison;
    hconcat(image, filtered, comparison);

    // Display comparison
    namedWindow("Comparison: Original (Left) vs Filtered (Right)", WINDOW_NORMAL);
    imshow("Comparison: Original (Left) vs Filtered (Right)", comparison);
    cout << "Press any key in the window to continue" << endl;
    waitKey(0);

    // Save output image
    if (imwrite(outputPath, filtered)) {
        cout << "Output saved to " << outputPath << endl;
    }
    else {
        cout << "Error: Could not save image to " << outputPath << endl;
        return -1;
    }

    return 0;
}