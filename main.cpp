//#include <QtCore/QCoreApplication>
/*#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <ml.h>
*/
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <time.h>
#include <set>
#include <math.h>
#include <string>
#include <iostream>
#include <iterator>
#include <QtGui>
#include "GCoptimization.h"
#include <algorithm>
//typedef float EnergyTermType;
//typedef double EnergyType;

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "GCoptimization.h"

using namespace std;
using namespace cv;
using namespace flann;
using namespace cvflann;

int topNum;
vector<Point2i> gcLabels;
vector<Point2i> gcNodes;
cv::Mat img_ycb;
cv::Mat mask;
map<pair<int,int>,int> maskPointToNodeIdx;

#define AMIR_MAX_NUM 100000

bool isValid( int newX, int newY)
{
        if( newX >= 0 && newY >= 0 && newX < mask.cols && newY < mask.rows && mask.at<uchar>(newY,newX) != 0 )
        {
                return true;
        }
        return false;
}

//TODO: Neighbor

bool isInside(int newX, int newY)
{
        if( newX >= 0 && newY >= 0 && newX < mask.cols && newY < mask.rows )
        {
                return true;
        }
        return false;

}


int smoothFn(int p1, int p2, int l1, int l2)
{

        if(l1 == l2)
        {
            return 0;
        }

        int retMe = 0;

        Point2i x1_s_a = gcLabels[l1] + gcNodes[p1];
        Point2i x2_s_b = gcLabels[l2] + gcNodes[p2];

        if( isValid(x1_s_a.x, x1_s_a.y) && isValid(x2_s_b.x, x2_s_b.y) )
        {
                Point2i x1_s_b = gcNodes[p1] + gcLabels[l2];
                Point2i x2_s_a = gcNodes[p2] + gcLabels[l1];

                if( isValid( x1_s_b.x, x1_s_b.y ) && isValid(x2_s_a.x, x2_s_a.y) )
                {
                        Vec3b v1_a = img_ycb.at<Vec3b>( x1_s_a.y, x1_s_a.x);
                        Vec3b v1_b = img_ycb.at<Vec3b>( x1_s_b.y, x1_s_b.x);
                        Vec3b diff1 = v1_a - v1_b;

                        retMe += diff1[0]*diff1[0] + diff1[1]*diff1[1] + diff1[2]*diff1[2];


                        Vec3b v2_a = img_ycb.at<Vec3b>( x2_s_a.y, x2_s_a.x);
                        Vec3b v2_b = img_ycb.at<Vec3b>( x2_s_b.y, x2_s_b.x);
                        Vec3b diff2 = v2_a - v2_b;

                        retMe += diff2[0]*diff2[0] + diff2[1]*diff2[1] + diff2[2]*diff2[2];
                }
                else
                {
                    return AMIR_MAX_NUM;
                }

                return retMe;
        }

        return AMIR_MAX_NUM;

}

int dataFn(int p, int l, void *data)
{
        int newX = gcLabels[l].x + gcNodes[p].x;
        int newY = gcLabels[l].y + gcNodes[p].y;
        if( isValid(newX,newY) )
        {
                return 0;
        }
        return AMIR_MAX_NUM;
}



cv::Mat imageToFeatureVec( cv::Mat img, int width, map<int,pair<int,int> >& idxToPointMap)
{
        int nch = img.channels();
        int img_w = img.cols;
        int img_h = img.rows;
        int w_2 = width*width;
        int f_vec_len = nch * w_2;
        int num_vec = img_w * img_h / w_2;
        cerr<<"[nch: "<<nch<<" ,f_vec_len: "<<f_vec_len<<" ,num_vec: "<<num_vec<<" ]"<<endl;
        cv::Mat retMe(num_vec,f_vec_len,CV_32FC1);
        for(int y = 0; y < img_h; y++)
        {
                for(int x = 0; x < img_w; x++)
                {
                        int vec_i = (y/width)*(img_w/width) + x/width;

                        for(int c = 0; c < nch; c++)
                        {
                                int f_vec_i = nch * ((y%width)*width + (x%width)) + c;
                                retMe.at<float>(vec_i,f_vec_i) = (float)img.at<cv::Vec3b>(y,x)[c];
                        }

                        if( (x%width == 0) && (y%width == 0) )
                        {
                                idxToPointMap[vec_i] = make_pair(x,y);
                        }

                }
        }
        return retMe;
}


void sortPoints(vector<Point2i>& points, vector<double>& values)
{
        /*for(int i = 0; i < points.size(); i++)
        {
                cout<<"Value: "<<values[i]<<", Point: "<<points[i].x<<", "<<points[i].y<<endl;
        }
        cout<<" *****\n*****\n ******\n****\n*****\n*******\n*******\n*******\n*****\n";
*/

        for(int i =  points.size() -1; i > 0; i--)
        {
                for(int j = 0; j < i; j++)
                {
                        if( values[j] < values[j+1])
                        {
                                double tmpVal = values[j];
                                Point2i tmpPoint = points[j];
                                values[j] = values[j+1];
                                points[j] = points[j+1];
                                points[j+1] = tmpPoint;
                                values[j+1] = tmpVal;
                        }
                }
        }
        /*for(int i = 0; i < points.size(); i++)
        {
                cout<<"Value: "<<values[i]<<", Point: "<<points[i].x<<", "<<points[i].y<<endl;
        }*/
}


vector<Point2i> getTopDisplacemetns( cv::Mat hist, int width, double sigma)
{
        int img_w = hist.cols;
        int img_h = hist.rows;
        cv::GaussianBlur(hist, hist, Size(0,0), sigma);

        vector<Point2i> points;
        vector<double> values;
        for(int y = 0; y < img_h; y+=width)
        {
                for(int x = 0; x < img_w; x+=width)
                {
                        if( (x + width) < hist.cols && (y+width) < hist.rows )
                        {
                                Mat C = hist(Range(y,y+width), Range(x,x+width));
                                double minVal,maxVal;
                                Point2i minLoc, maxLoc;
                                cv::minMaxLoc( C, &minVal, &maxVal, &minLoc, &maxLoc );
                                points.push_back( Point2i(x + maxLoc.x - hist.cols/2,y+ maxLoc.y - hist.rows/2));
                                values.push_back( maxVal);
                        }

                }
        }

        sortPoints( points, values);
        return points;
}

cv::Mat imageToFeatureVecCoarse( cv::Mat img, cv::Mat mask, int width, map<int,pair<int,int> >& idxToPointMap)
{
        int nch = img.channels();
        int img_w = img.cols;
        int img_h = img.rows;
        int w_2 = width*width;
        int f_vec_len = nch * w_2;
        int num_vec = (img_w - width) * (img_h - width);
        cerr<<"[nch: "<<nch<<" ,f_vec_len: "<<f_vec_len<<" ,num_vec: "<<num_vec<<" ]"<<endl;
        cv::Mat retMe(num_vec,f_vec_len,CV_32FC1);
        int i = 0;
        for(int y = 0; y < img_h - width; y++)
        {
                for(int x = 0; x < img_w - width; x++)
                {
                        //int vec_i = (y/width)*(img_w/width) + x/width;
                        Mat C = img(Range(y,y+width),Range(x,x+width)).clone();
                        Mat mask_C = mask(Range(y,y+width), Range(x,x+width));

                        Mat n = C.reshape(1, 1).clone();
                        if( cv::countNonZero( mask_C) == width * width )
                        {
                                for(int j = 0; j < f_vec_len; j++)
                                {
                                        retMe.at<float>(i,j) = (float) n.at<uchar>(0,j);
                                }

                                idxToPointMap[i]  = make_pair(x,y);

                                i++;
                        }

                }
        }
        cerr<<"[Reduced Num: "<<i<<" ]"<<endl;

        cv::Mat reducedRetMe = retMe(Range(0,i),Range(0,f_vec_len)).clone();

        return reducedRetMe;
}

vector<Point2i> maskToNodes(cv::Mat mask)
{
        vector<Point2i> nodes;

        for(int y = 0; y < mask.rows; y++)
        {
                for(int x = 0; x < mask.cols; x++)
                {
                        if( mask.at<uchar>(y, x) == 0)
                        {
                                nodes.push_back(Point2i(x,y));
                                maskPointToNodeIdx[make_pair(x,y)] = nodes.size()-1;
                        }
                        else
                        {
                                maskPointToNodeIdx[make_pair(x,y)] = -1;
                        }
                }
        }
        return nodes;
}

Mat rand_select_and_shuffle(Mat in, int num_samples)
{
	if ( num_samples > in.rows)
	{
		num_samples = in.rows;
	}

	int total = in.rows;
	int *idx = new int[total];
	for(int i = 0; i < total; i++)
		idx[i] = i;
	srand(time(0));
	random_shuffle(&idx[0], &idx[total-1]);

	Mat retMe;
	retMe.create( num_samples, in.cols, in.type() );

	for(int i = 0; i < num_samples; i++)
	{
		in.row(idx[i]).copyTo( retMe.row(i) );
	}

	return retMe;
}

int main()
{
        int knn = 25;
        float tau = 16;
        int n_kdd = 1;
        int width = 8;
        int d = 8;
        topNum = 60;

        char* img_path = "img.PNG";
		//"/Users/amirrahimi/Downloads/Applications/cvpr10Data/images/img-op1-p-251t000-resized.jpg";
        char* mask_path = "mask.bmp";
		//"/Users/amirrahimi/Desktop/TMP/OccMasks/img-op1-p-251t000_mask.png";
        //char* img_path = "/Users/amirrahimi/Desktop/Picture2.png";
        cv::Mat img = cv::imread(img_path);
        mask = cv::imread(mask_path, 0 );
        //cv::imshow("salam",img);
        //cv::waitKey(0);

        cv::cvtColor(img,img_ycb,CV_BGR2YCrCb);
        double minVal,maxVal;
        map<int,pair<int,int> > idxToPointMap;

        cv::Mat trainMe = imageToFeatureVecCoarse( img_ycb, mask, width, idxToPointMap);
	//trainMe = rand_select_and_shuffle( trainMe, trainMe.rows/10 );

        cout<<"Rows: "<<trainMe.rows<<" Cols: "<<trainMe.cols<<endl;
        cv::Mat hist = Mat::zeros( img.rows*2, img.cols*2, CV_32FC1);
        flann::KDTreeIndexParams params(n_kdd);
        //flann::Index_<float> index(a,params);
        cout<<"Training..."<<endl;
        flann::Index index(trainMe,params);
        cout<<"Querying..."<<endl;
        int j = 600;//15*8;
	for( int j = 0; j < trainMe.rows; j++)
        {
                cv::Mat canvas = img.clone();
                cv::Mat qMat = trainMe.row(j);
                /*vector<float> q;
                  q.push_back(-4.0f);
                  q.push_back(-4.0f);*/
                vector<int> indices;
                vector<float> dists;
                for(int i =0; i < knn; ++i)
                {
                        indices.push_back(-1);
                        dists.push_back(-1.0f);
                }
                index.knnSearch(qMat,indices,dists,knn);

                int xx0 = idxToPointMap[j].first;
                int yy0 = idxToPointMap[j].second;
                /*rectangle(canvas, Point(xx0, yy0),
                                Point(xx0+width, yy0+width),
                                Scalar(0,255,0));*/

                for(int i = 1; i < knn; ++i)
                {
                        //cout<<indices[i]<<" , dist: "<<dists[i]<< ", ( "<<idxToPointMap[indices[i]].first<<","<<idxToPointMap[indices[i]].second<<")"<<endl;
                        int xx = idxToPointMap[indices[i]].first;
                        int yy = idxToPointMap[indices[i]].second;
                        if( abs(xx-xx0) + abs(yy-yy0) > tau )
                        {
                                /*rectangle(canvas, Point( xx, yy),
                                                Point(xx+width, yy+width),
                                                Scalar(0,0,255));*/
                                hist.at<float>( img.rows + yy - yy0, img.cols + xx - xx0) += 1;
                                //hist.at<float>( img.rows + yy0 - yy, img.cols + xx0 - xx) += 1;
                                break;
                        }

                }
                //cv::imshow("matchings",canvas);
                //cv::waitKey(0);
        }

        cv::minMaxLoc(hist,&minVal,&maxVal);
        cv::imshow("hist", hist/maxVal);
        cv::waitKey(0);
        vector<Point2i> tops = getTopDisplacemetns( hist, d, sqrt(2.0));
        for(int _t = 0; _t < topNum; _t++)
        {
                cerr<<"("<<tops[_t].x<<", "<<tops[_t].y<<")"<<endl;
        }

        gcLabels = tops;
        gcNodes =  maskToNodes( mask);

        try
        {
                GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(gcNodes.size(), topNum );
                gc->setDataCost( &dataFn, NULL);
                gc->setSmoothCost( &smoothFn );
                int aa = 0;
                for(int img_y = 0; img_y < mask.rows; img_y++)
                {
                        for(int img_x = 1; img_x < mask.cols; img_x++)
                        {
                                if( maskPointToNodeIdx[make_pair(img_x,img_y)] >= 0 && maskPointToNodeIdx[make_pair(img_x-1,img_y)] >= 0 )
                                {
                                        gc->setNeighbors(maskPointToNodeIdx[make_pair(img_x,img_y)], maskPointToNodeIdx[make_pair(img_x -1,img_y) ],1 );
                                        aa++;
                                }
                        }
                }

                for(int img_y = 1; img_y < mask.rows; img_y++)
                {
                        for(int img_x = 0; img_x < mask.cols; img_x++)
                        {
                                if( maskPointToNodeIdx[make_pair(img_x,img_y)] >= 0 && maskPointToNodeIdx[make_pair(img_x,img_y-1)] >= 0 )
                                {
                                        gc->setNeighbors(maskPointToNodeIdx[make_pair(img_x,img_y)], maskPointToNodeIdx[make_pair(img_x,img_y-1) ] ,1);
                                        aa++;
                                }
                        }
                }

                cout<<"AA: "<<aa<<endl;
                cout<<"Energy Before: "<<gc->compute_energy()<<endl;
                for(int lbl = 0; lbl <=2; lbl++)
                {
                    for(int lbl2 = 0;lbl2<=2;lbl2++)
                    {
                        cout<<"\t*** lbl: "<<lbl<<", lbl2: "<<lbl2<<"***"<<endl;
                        int s_0 = smoothFn(6447, 6378, lbl, lbl);
                        int s_1 = smoothFn(6447, 6378, lbl2, lbl2);
                        int s_2 = smoothFn(6447, 6378, lbl, lbl2);
                        int s_3 = smoothFn(6447, 6378, lbl2, lbl);
                        int d_1 = dataFn(6447,lbl2,NULL);
                        int d_2 = dataFn(6378,lbl2,NULL);
                        int d_3 = dataFn(6447,lbl,NULL);
                        int d_4 = dataFn(6378,lbl,NULL);
                        //int d_5 = dataFn(6696,2,NULL);
                        //int d_6 = dataFn(6695,2,NULL);
                        cout<<"s_0: "<<s_0<<", s_1: "<<s_1<<", s_2: "<<s_2<<", s_3: "<<s_3;
                        cout<<", d_1: "<<d_1<<", d_4: "<<d_4<<", d_2: "<<d_2<<", d_3: "<<d_3<<endl;
                    }
                }


                gc->swap(2);
                cout<<"Energy After 1: "<<gc->compute_energy()<<endl;
		gc->swap(2);
		cout<<"Energy After 2: "<<gc->compute_energy()<<endl;
		//gc->swap(60);
		//cout<<"Energy After 3: "<<gc->compute_energy()<<endl;



                for(int _i = 0; _i < gcNodes.size(); _i++)
                {
                    int res = gc->whatLabel(_i);
                    for(int _nch = 0; _nch < 3; _nch++)
                    img.at<Vec3b>(gcNodes[_i])[_nch] = img.at<Vec3b>(gcNodes[_i]+gcLabels[res])[_nch];
                }

                imshow("completed",img);
                waitKey(0);

        }catch(GCException e)
        {
                e.Report();
        }
        return 0;
}
