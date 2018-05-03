#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
SequenceDataLayer<Dtype>:: ~SequenceDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void SequenceDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int num_frames  = this->layer_param_.sequence_data_param().num_frames();
	const int num_segments = this->layer_param_.sequence_data_param().num_segments();
    const int num_shots = this->layer_param_.sequence_data_param().num_shots();
    const int feature_size = this->layer_param_.sequence_data_param().feature_size();
//    const int d_bytes = this->layer_param_.sequence_data_param().d_bytes();
    // full path to features
	const string& feature_source = this->layer_param_.sequence_data_param().feature_source();
	// file included relative paths to features, #of frames, labels
	const string& feature_description = this->layer_param_.sequence_data_param().feature_description();

	// loading video files (path, duration, label)
        LOG(INFO) << "Opening features description file: " << feature_description;
    std::ifstream infile(feature_description.c_str());
    string line;

	while(getline(infile, line)) {
		std::istringstream iss(line);
		string filename;
		vector<int> labels;
		int label;
		int length;
		iss >> filename >> length;
		while(iss >> label) {
			labels.push_back(label);
		}
		string full_path = feature_source + "/" + filename;
		lines_.push_back(std::make_pair(full_path, labels));
		lines_duration_.push_back(length);
	}

	// TODO: uniform sampling from sequence, like making shots
	int uniform_sampling = 5;
	vector<std::pair<int, int>> tmp_vec;
	int start = 0;
	int end = 0;
	int step = 0;
	for(int vid = 0; vid < lines_duration_.size(); ++vid) {
		step = lines_duration_[vid] / uniform_sampling;
		end = step;
		for(int shot = 0; shot < uniform_sampling; ++shot) {
			if(start != 0) {
				start += step;
				end = start + step;
			}
			if(this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_RGB)
				end -= 1;
			if(this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_FLOW)
				end -= 2;
			if(end - start + 1 > num_frames)
				tmp_vec.push_back(std::make_pair(start, end));
		}
		if(this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_RGB)
			end = lines_duration_[vid];
		if(this->layer_param_.sequence_data_param().modality() == SequenceDataParameter_Modality_FLOW)
			end = lines_duration_[vid] - 1;
		if(end - start + 1 > num_frames)
			tmp_vec.push_back(std::make_pair(start, end));
		lines_shot_.push_back(tmp_vec);
	}

	// TODO: shuffle

	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	LOG(INFO) << "A total of " << lines_shot_.size() << " samples.";

	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));

	Datum datum;
	lines_id_ = 0;
	vector<std::pair<int, int> > cur_shot_list = lines_shot_[lines_id_];
	vector<int> offsets;

	// TODO: choose random label
	int label = lines_[lines_id_].second[0];
	CHECK(ReadFeaturesToDatum(lines_[lines_id_].first, label, 0, feature_size, num_frames, &datum));

	const int batch_size = this->layer_param_.sequence_data_param().batch_size();
	top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());

	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height()
			<< "," << top[0]->width();

	top[1]->Reshape(batch_size, 1, 1, 1);
	this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

//	vector<int> label_shape(1, batch_size);
//	top[1]->Reshape(label_shape);
//	this->prefetch_label_.Reshape(label_shape);

	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	this->transformed_data_.Reshape(top_shape);
}

template <typename Dtype>
void SequenceDataLayer<Dtype>::ShuffleSequences(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
        caffe::rng_t* prefetch_rng3 = static_cast<caffe::rng_t*>(prefetch_rng_3_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(), prefetch_rng2);
	shuffle(lines_shot_.begin(), lines_shot_.end(), prefetch_rng3);
}

template <typename Dtype>
void SequenceDataLayer<Dtype>::InternalThreadEntry(){

	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	SequenceDataParameter sequence_data_param = this->layer_param_.sequence_data_param();
	const int batch_size = sequence_data_param.batch_size();
	const int feature_size = sequence_data_param.feature_size();
//	const int d_bytes = sequence_data_param.d_bytes();
	const int num_frames = sequence_data_param.num_frames();
	const int num_segments = sequence_data_param.num_segments();
	const int num_shots = sequence_data_param.num_shots();

	const int lines_size = lines_.size();

	for(int item_id = 0; item_id < batch_size; ++item_id) {
		CHECK_GT(lines_size, lines_id_);

		int label = lines_[lines_id_].second[0];
		ReadFeaturesToDatum(lines_[lines_id_].first, label, 0, feature_size, num_frames, &datum);

		int offset1 = this->prefetch_data_.offset(item_id);
		this->transformed_data_.set_cpu_data(top_data + offset1);
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
		top_label[item_id] = lines_[lines_id_].second[0];

		++lines_id_;
		if(lines_id_ >= lines_size) {
			lines_id_ = 0;
		}

	}
}

INSTANTIATE_CLASS(SequenceDataLayer);
REGISTER_LAYER_CLASS(SequenceData);
}
