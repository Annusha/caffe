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
#include "caffe/util/benchmark.hpp"

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
//		LOG(INFO) << "file: " << iss;
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
	for(int vid = 0; vid < lines_duration_.size(); ++vid) {
		vector<std::pair<int, int> > tmp_vec;
		int start = 0;
		int end = 0;
		int step = 0;
		int lenn = lines_duration_[vid];
		step = (int) lines_duration_[vid] / uniform_sampling;
		end = step;
		for(int shot = 0; shot < uniform_sampling; ++shot) {
			end -= 1;
			if(end - start + 1 > num_frames) {
				tmp_vec.push_back(std::make_pair(start, end));
			}
			start += step;
			end = start + step;
		}
		end = lines_duration_[vid] - 1;
		if(end - start + 1 > num_frames) {
			tmp_vec.push_back(std::make_pair(start, end));
		}
		lines_shot_.push_back(tmp_vec);
	}

	// TODO: shuffle
	if (this->layer_param_.sequence_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_3_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleSequences();
}

	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	LOG(INFO) << "A total of " << lines_shot_.size() << " shots.";
//	for (int i = 0; i < lines_shot_.size(); ++i)
//		LOG(INFO) << lines_[i].first << " " << lines_shot_[i].size();

	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));

	Datum datum;
	lines_id_ = 0;
	vector<std::pair<int, int> > cur_shot_list = lines_shot_[lines_id_];
	vector<int> offsets;

	for(int i = 0; i < num_shots; ++i) {
		int shot_idx = i;
		if(i >= cur_shot_list.size()) {
			caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
			shot_idx = (*frame_rng)() % (cur_shot_list.size());
		}
		int start_idx = cur_shot_list[shot_idx].first;
		int end_idx = cur_shot_list[shot_idx].second;
		int average_duration = (int) (end_idx - start_idx + 1) / num_segments;
		for(int j = 0; j < num_segments; ++j) {
			if(average_duration < num_frames) {
				offsets.push_back(start_idx);
				continue;
			}
			caffe::rng_t* frame_rng1 = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
			int offset = (*frame_rng1)() % (average_duration - num_frames + 1);
//			offsets.push_back(0);
			offsets.push_back(start_idx + offset + j * average_duration);
		}
	}


	// TODO: choose random label
	caffe::rng_t* label_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
	int label_idx = (*label_rng)() % (lines_[lines_id_].second.size());
	int label = lines_[lines_id_].second[label_idx];

	LOG(INFO) << lines_[lines_id_].first << " " << lines_duration_[lines_id_];
	CHECK(ReadFeaturesToDatum(lines_[lines_id_].first, label, offsets, feature_size, num_frames, &datum));

	const int batch_size = this->layer_param_.sequence_data_param().batch_size();
	top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());

	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height()
			<< "," << top[0]->width();

	top[1]->Reshape(batch_size, 1, 1, 1);
	this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

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
	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;

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
		timer.Start();
		CHECK_GT(lines_size, lines_id_);

		vector<std::pair<int, int> > cur_shot_list = lines_shot_[lines_id_];
		vector<int> offsets;
		int lenn = lines_duration_[lines_id_];
		for(int i = 0; i < num_shots; ++i) {
			int shot_idx = i;
			if(i >= cur_shot_list.size()) {
				caffe::rng_t* frame_rng1 = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
				shot_idx = (*frame_rng1)() % (cur_shot_list.size());
			}
			int start_idx = cur_shot_list[shot_idx].first;
			int end_idx = cur_shot_list[shot_idx].second;
			int average_duration = (int) (end_idx - start_idx + 1) / num_segments;
			for(int j = 0; j < num_segments; ++j) {
				if(average_duration < num_frames) {
					offsets.push_back(start_idx);
					continue;
				}
				caffe::rng_t* frame_rng2 = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
				int offset = (*frame_rng2)() % (average_duration - num_frames + 1);
				int l = start_idx + offset + j * average_duration;
				if(l == lenn) {
					DLOG(INFO) << "bad offset ";
				}
				offsets.push_back(start_idx + offset + j * average_duration);
			}
		}

		caffe::rng_t* label_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		int label_idx = (*label_rng)() % (lines_[lines_id_].second.size());
		int label = lines_[lines_id_].second[label_idx];
		ReadFeaturesToDatum(lines_[lines_id_].first, label, offsets, feature_size, num_frames, &datum);

		read_time += timer.MicroSeconds();
		timer.Start();
		int size_of = offsets.size();
		int offset1 = this->prefetch_data_.offset(item_id);
		this->transformed_data_.set_cpu_data(top_data + offset1);
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
		trans_time += timer.MicroSeconds();
		top_label[item_id] = lines_[lines_id_].second[label_idx];

		++lines_id_;
		if(lines_id_ >= lines_size) {
			lines_id_ = 0;
			if(this->layer_param_.sequence_data_param().shuffle()){
				ShuffleSequences();
			}
		}

	}

	batch_timer.Stop();
//	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
//	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(SequenceDataLayer);
REGISTER_LAYER_CLASS(SequenceData);
}
