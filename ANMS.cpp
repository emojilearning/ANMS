#include <opencv2/opencv.hpp>
#include <time.h>

using namespace cv;
using namespace std;

double computeR(Point2i x1, Point2i x2)
{
	return norm(x1 - x2);
}
template <typename T>
vector<size_t>  sort_indexes(const vector<T>  & v) {

	// initialize original index locations
	vector<int> idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });

	return idx;
}

vector<KeyPoint> ANMS(const std::vector<KeyPoint>& kpts, int num = 500)
{
	int sz = kpts.size();
	double maxmum = 0;
	vector<double> roblocalmax(kpts.size());
	vector<double> raduis(kpts.size(), INFINITY);
	for (size_t i = 0; i < sz; i++)
	{
		auto rp = kpts[i].response;
		if (rp > maxmum)
			maxmum = rp;
		roblocalmax[i] = rp*0.9;
	}
	auto max_response = maxmum*0.9;
	for (size_t i = 0; i < sz; i++)
	{
		double rep = kpts[i].response;
		Point2i p = kpts[i].pt;
		auto& rd = raduis[i];
		if (rep > max_response)
		{
			rd = INFINITY;
		}
		else
		{
			for (size_t j = 0; j < sz; j++)
			{
				if (roblocalmax[j] > rep)
				{
					auto d = computeR(kpts[j].pt, p);
					if (rd > d)
						rd = d;
				}
			}
		}
	}
	auto sorted = sort_indexes(raduis);
	vector<KeyPoint> rpts;

	for (size_t i = 0; i < num; i++)
	{
		rpts.push_back(kpts[sorted[i]]);
	}
	return std::move(rpts);
}


void Dilate(const Mat& src, Mat& dstmat,int ksize) {
	//row and col is inversed here,but I'll modify it later...Maybe
	const int half_size = ksize / 2;
	const int height = src.rows;
	const int width = src.cols;
	auto srs = src.rows - ksize;
	auto crs = src.cols - ksize;
	auto dst = dstmat.ptr<float>(0);
	for (int r = 0; r < srs; r++)
	{
		const float* pvalue = src.ptr<float>(r + half_size,0);
		for (int c = 0; c < crs; c++)
		{
			float value = pvalue[c + 3];
			if (value == 0)
				continue;
			for (size_t i = 0; i < ksize; i++)
			{
				for (size_t j = 0; j < ksize; j++)
				{
					auto& val = dst[(r + j)*width +c+i];
					val = value> val ? value : val;
				}
			}
		}
	}
}

float kpgraph[640][480];
float fgraph[640][480];
vector<KeyPoint> NMS(const std::vector<KeyPoint>& kpts, int iteration_num = 1, int ksize = 9)
{
	//cout << kpts.size() << endl;
	fill(&kpgraph[0][0], &kpgraph[0][0] + 640 * 480, 0);
	fill(&fgraph[0][0], &fgraph[0][0] + 640 * 480, 0);
	const int w_size = 31;
	int sz = kpts.size();
	for (size_t i = 0; i < sz; i++)
	{
		kpgraph[(int)kpts[i].pt.x][(int)kpts[i].pt.y] = kpts[i].response;
	}
	Mat kgrp(640, 480, CV_32F, &kpgraph[0][0]);
	Mat fgrp(640, 480, CV_32F, &fgraph[0][0]);

	Dilate(kgrp, fgrp, ksize);
	for (size_t i = 0; i < iteration_num - 1; i++)
	{
		Dilate(fgrp.clone(), fgrp, ksize);
	}
	//imshow("1", fgrp);
	//imshow("0", kgrp);
	//waitKey(0);
	vector<KeyPoint> rkpts;
	for (size_t i = 0; i < sz; i++)
	{
		if (kpts[i].response*1.1 > fgraph[(int)kpts[i].pt.x][(int)kpts[i].pt.y])
		{
			rkpts.push_back(kpts[i]);
		}
	}
	//cout << rkpts.size() << endl;
	return std::move(rkpts);
}


int main()
{
	auto img = imread("../680.jpg");
	std::vector<KeyPoint> kpts;
	kpts.reserve(20000);
	Ptr<ORB> orb_detector = ORB::create(9000, 1.2, 8, 16, 0, 2, 0, 31, 5);
	clock_t acc = 0;
	Mat des;
	orb_detector->detect(img, kpts);

	auto t1 = clock();
	auto r = NMS(kpts,1,9);
	auto t2 = clock();

	acc += t2 - t1;
	Mat out;
	cout << kpts.size() << endl;
	cout << r.size() << endl;
	std::cout << acc << std::endl;
	drawKeypoints(img, r, out);
	imshow("null", out);
	orb_detector = ORB::create(3000, 1.2, 8, 16, 0, 2, 0, 31, 5);
	orb_detector->detect(img, kpts);
	drawKeypoints(img, kpts, out);
	imshow("2", out);
	waitKey(0);
}
