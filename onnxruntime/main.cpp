#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
};

typedef struct corner_points
{
	Point pt1;
	Point pt2;
	Point pt3;
	Point pt4;
} corner_points;

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
	corner_points four_points;
} BoxInfo;

class YOLO
{
public:
	YOLO(Net_config config);
	void detect(Mat& frame);
private:
	const float anchors[3][6] = { {4,5,  8,10,  13,16}, {23,29,  43,55,  73,105},{146,217,  231,300,  335,433} };
	const float stride[3] = { 8.0, 16.0, 32.0 };
	int inpWidth;
	int inpHeight;
	int nout;
	float confThreshold;
	float nmsThreshold;
	float objThreshold;

	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
	vector<float> input_image_;
	void normalize_(Mat img);
	void nms(vector<BoxInfo>& input_boxes);
	const bool keep_ratio = true;
	void sigmoid(Mat* out, int length);
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

static inline float sigmoid_x(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

YOLO::YOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;

	string model_path = "best.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];
}

Mat YOLO::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLO::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
				
			}
		}
	}
}

void YOLO::sigmoid(Mat* out, int length)
{
	float* pdata = (float*)(out->data);
	int i = 0;
	for (i = 0; i < length; i++)
	{
		pdata[i] = 1.0 / (1 + expf(-pdata[i]));
	}
}

void YOLO::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void YOLO::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	const float* preds = ort_outputs[0].GetTensorMutableData<float>();

	/////generate proposals
	vector<BoxInfo> generate_boxes;
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	int n = 0, q = 0, i = 0, j = 0, k = 0; ///xmin,ymin,xamx,ymax,box_score,x1,y1, ... ,x4,y4,plate_score
	for (n = 0; n < 3; n++)   ///特征图尺度
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		for (q = 0; q < 3; q++)    ///anchor
		{
			const float anchor_w = this->anchors[n][q * 2];
			const float anchor_h = this->anchors[n][q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float box_score = sigmoid_x(preds[4]);
					if (box_score > this->objThreshold)
					{
						float plate_score = sigmoid_x(preds[13]);
						//if (plate_score > this->confThreshold)
						//{ 
						float cx = (sigmoid_x(preds[0]) * 2.f - 0.5f + j) * this->stride[n];  ///cx
						float cy = (sigmoid_x(preds[1]) * 2.f - 0.5f + i) * this->stride[n];   ///cy
						float w = powf(sigmoid_x(preds[2]) * 2.f, 2.f) * anchor_w;   ///w
						float h = powf(sigmoid_x(preds[3]) * 2.f, 2.f) * anchor_h;  ///h

						float xmin = (cx - padw - 0.5 * w)*ratiow;
						float ymin = (cy - padh - 0.5 * h)*ratioh;
						float xmax = (cx - padw + 0.5 * w)*ratiow;
						float ymax = (cy - padh + 0.5 * h)*ratioh;

						corner_points pts;
						k = 5;
						pts.pt1.x = int((preds[k] * anchor_w + j * this->stride[n] - padw) * ratiow);
						pts.pt1.y = int((preds[k + 1] * anchor_h + i * this->stride[n] - padh) * ratioh);
						k = 7;
						pts.pt2.x = int((preds[k] * anchor_w + j * this->stride[n] - padw) * ratiow);
						pts.pt2.y = int((preds[k + 1] * anchor_h + i * this->stride[n] - padh) * ratioh);
						k = 9;
						pts.pt3.x = int((preds[k] * anchor_w + j * this->stride[n] - padw) * ratiow);
						pts.pt3.y = int((preds[k + 1] * anchor_h + i * this->stride[n] - padh) * ratioh);
						k = 11;
						pts.pt4.x = int((preds[k] * anchor_w + j * this->stride[n] - padw) * ratiow);
						pts.pt4.y = int((preds[k + 1] * anchor_h + i * this->stride[n] - padh) * ratioh);

						generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, plate_score, 0, pts});
						//}
					}
					preds += nout;
				}
			}
		}
	}

	nms(generate_boxes);
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		rectangle(frame, Point(int(generate_boxes[i].x1), int(generate_boxes[i].y1)), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
		circle(frame, generate_boxes[i].four_points.pt1, 2, Scalar(0, 255, 0), -1);
		circle(frame, generate_boxes[i].four_points.pt2, 2, Scalar(0, 255, 0), -1);
		circle(frame, generate_boxes[i].four_points.pt3, 2, Scalar(0, 255, 0), -1);
		circle(frame, generate_boxes[i].four_points.pt4, 2, Scalar(0, 255, 0), -1);
	}
}

int main()
{
	Net_config yolo_nets = { 0.3, 0.5, 0.3};
	YOLO yolo_model(yolo_nets);
	string imgpath = "imgs/1.jpg";
	Mat srcimg = imread(imgpath);
	yolo_model.detect(srcimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}
