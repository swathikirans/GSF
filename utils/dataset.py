import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import sys
import h5py
import io
import gulpio2

class VideoRecord(object):
    def __init__(self, row, multilabel):
        self._data = row
        self._multilabel = multilabel

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def label_verb(self):
        if self._multilabel:
            return int(self._data[3])
        else:
            return 0

    @property
    def label_noun(self):
        if self._multilabel:
            return int(self._data[4])
        else:
            return 0

    @property
    def start_frame(self):
        if self._multilabel:
            return int(self._data[5])
        else:
            return 0


class VideoDataset(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, num_clips=1,
                 load_from_video=False, frame_interval=5,
                 sparse_sampling=True, multilabel=False, dense_sample=False, from_hdf5=False, from_gulp=False, mode="train", 
                 random_shuffling=False,):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_clips = num_clips
        self.load_from_video = load_from_video
        self.frame_interval = frame_interval
        self.sparse_sampling = sparse_sampling
        self.multilabel = multilabel
        self.dense_sample = dense_sample
        self.from_hdf5 = from_hdf5
        self.from_gulp = from_gulp
        self.h5_file = None
        if self.from_gulp and self.multilabel:
            self.root_path = self.root_path.replace("frames", "frames_gulp")
        self.mode = mode
        self.random_shuffling = random_shuffling

        self._parse_list()

    def _load_image(self, directory, idx):
        try:
            return [
                Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        except Exception:
            print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]

    def _load_from_video(self, path, frame_ids):
        return 0

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        if self.from_gulp:
            self.list_file = self.list_file.replace(".txt", "_gulp.txt")
        if "kinetics" in self.root_path:
            tmp = [x.strip().split(',') for x in open(self.list_file)]
        else:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>=3]
        self.video_list = [VideoRecord(item, self.multilabel) for item in tmp]
        if self.from_gulp:
            mode = "train" if "val" not in self.list_file else "val"
            self.gulp = gulpio2.GulpDirectory(os.path.join(self.root_path, mode))
            len_gulp = 0
            for dict in self.gulp.all_meta_dicts:
                len_gulp += len(dict)
            assert len_gulp == len(self.video_list), f"No. of samples is different {self.list_file}({len(self.video_list)}) | {os.path.join(self.root_path, mode)}({len_gulp})"
        print('video number:%d'%(len(self.video_list)))

    def _sample_indices(self, record=None, num_frames=None):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            num_frames = record.num_frames if record is not None else num_frames

            average_duration = num_frames // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments) + 1
            elif num_frames > self.num_segments:
                offsets = np.sort(randint(num_frames, size=self.num_segments)) + 1
            else:
                tick = num_frames / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
                # offsets = np.zeros((self.num_segments,))
                # offsets = np.concatenate(
                #     [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        return offsets

    def _sample_indices_dense(self, record):
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        return np.array(offsets) + 1

    def _sample_indices_video(self, record):

        """

        :param record: VideoRecord
        :return: list
        """
        num_frames = record.num_frames

        if not self.sparse_sampling:
            max_frame_ind = num_frames - (self.frame_interval * (self.num_segments-1)) - 1
            if max_frame_ind > 0:
                start_frame_ind = randint(max_frame_ind, size=1)[0]
                offsets = [start_frame_ind + (self.frame_interval * x) for x in range(self.num_segments)]
                # print(offsets)
            else:
                average_duration = num_frames // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                      size=self.num_segments)
                elif num_frames > self.num_segments:
                    offsets = np.sort(randint(num_frames, size=self.num_segments))
                else:
                    offsets = np.concatenate(
                        [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        else:
            average_duration = num_frames // self.num_segments
            if average_duration > 0:
                offsets = list(np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments))
            elif num_frames > self.num_segments:
                offsets = np.sort(randint(num_frames, size=self.num_segments))
            else:
                # offsets = np.zeros((self.num_segments,))
                offsets = np.concatenate(
                    [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
            return offsets + 1
        return np.array(offsets) + 1

    def _get_val_indices(self, record=None, num_frames=None):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            num_frames = record.num_frames if record is not None else num_frames
            # if num_frames > self.num_segments:
            #     tick = num_frames / float(self.num_segments)
            #     offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
            # else:
            #     tick = num_frames / float(self.num_segments)
            #     offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
            tick = num_frames / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
                # offsets = np.zeros((self.num_segments,))
                # offsets = np.concatenate(
                #     [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        return offsets

    def _get_val_indices_dense(self, record):
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        return np.array(offsets) + 1

    def _get_val_indices_video(self, record):
        num_frames = record.num_frames

        if not self.sparse_sampling:
            max_frame_ind = (num_frames-1) // 2 - ((self.frame_interval) * (self.num_segments // 2 - 1)) - 2
            if max_frame_ind > 0:
                start_frame_ind = max_frame_ind
                offsets = [start_frame_ind + (self.frame_interval * x) for x in range(self.num_segments)]
            else:
                if num_frames > self.num_segments:
                    tick = num_frames / float(self.num_segments)
                    offsets = [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
                else:
                    # offsets = np.zeros((self.num_segments,))
                    offsets = np.concatenate(
                        [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
        else:
            if num_frames > self.num_segments:
                tick = num_frames / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            offsets = offsets + 1
            offsets = list(offsets)
        return np.array(offsets) + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        num_frames = record.num_frames

        tick = num_frames / float(self.num_segments)

        if self.num_clips == 1:
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1

        elif self.num_clips == 2:
                offsets = [np.array([int(tick * x) for x in range(self.num_segments)]) + 1,
                           np.array([int(tick * x + tick / 2.0) for x in range(self.num_segments)]) + 1]
        elif self.num_clips == 10:
            offsets_clips = []
            for k in range(10):
                average_duration = num_frames // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                size=self.num_segments)
                elif num_frames > self.num_segments:
                    offsets = np.sort(randint(num_frames, size=self.num_segments))
                else:
                    offsets = np.concatenate(
                        [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)], axis=-1)
                offsets_clips.append(offsets+1)
            offsets = offsets_clips
        return offsets

    def _get_test_indices_dense(self, record):
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
        offsets = []
        for start_idx in start_list.tolist():
            offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        return np.array(offsets) + 1

    def _get_test_indices_video(self, record):
        
            num_frames = record.num_frames
            if not self.sparse_sampling:
                num_frames = num_frames - 1
                if self.num_clips == 1:
                    max_frame_ind = num_frames // 2 - ((self.frame_interval) * (self.num_segments // 2 - 1)) - 1
                    if max_frame_ind > 0:
                        start_frame_ind = max_frame_ind
                        offsets = [start_frame_ind + (self.frame_interval * x) for x in range(self.num_segments)]
                        # offsets = np.array(offsets) + 1
                    else:
                        if num_frames > self.num_segments:
                            tick = num_frames / float(self.num_segments)
                            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
                        else:
                            offsets = np.concatenate(
                                [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)],
                                axis=-1)
                    offsets = np.array(offsets) + 1
                else:
                    max_frame_ind = num_frames - (self.frame_interval * (self.num_segments-1)) - 1
                    if max_frame_ind > 0:
                        start_inds = np.linspace(1, max_frame_ind, self.num_clips)
                        offsets = []
                        for start_ind in start_inds:
                            offsets.append([int(start_ind) + (self.frame_interval * x) for x in range(self.num_segments)])
                    else:
                        if num_frames > self.num_segments:
                            tick = num_frames / float(self.num_segments)
                            offsets = [np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])] * self.num_clips
                        else:
                            offsets = [np.concatenate(
                                [np.arange(num_frames), np.ones(self.num_segments - num_frames) * (num_frames - 1)],
                                axis=-1)] * self.num_clips
                # offsets = offsets + 1

            else:
                tick = num_frames / float(self.num_segments)

                if self.num_clips == 1:
                    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1

                elif self.num_clips == 2:
                        offsets = [np.array([int(tick * x) for x in range(self.num_segments)]) + 1,
                                   np.array([int(tick * x + tick / 2.0) for x in range(self.num_segments)]) + 1]
        # print(offsets)
            return offsets



    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        # while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
        #     print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)))
        #     index = np.random.randint(len(self.video_list))
        #     record = self.video_list[index]
        if self.load_from_video:
            if not self.test_mode:
                segment_indices = self._sample_indices_video(record) if self.random_shift else self._get_val_indices_video(record)
            else:
                segment_indices = self._get_test_indices_video(record)
            return self.get_video(record, segment_indices)
        elif self.from_hdf5:
            if not self.sparse_sampling:
                if not self.test_mode:
                    segment_indices = self._sample_indices_video(record) if self.random_shift else self._get_val_indices_video(record)
                else:
                    segment_indices = self._get_test_indices_video(record)
            else:
                if not self.test_mode:
                    segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                else:
                    segment_indices = self._get_test_indices(record)
            if self.random_shuffling:
                segment_indices = np.random.permutation(segment_indices)
            return self.get_from_hdf5(record, segment_indices)
        elif self.from_gulp:
            if not self.sparse_sampling:
                if not self.test_mode:
                    segment_indices = self._sample_indices_video(record) if self.random_shift else self._get_val_indices_video(record)
                else:
                    segment_indices = self._get_test_indices_video(record)
            else:
                if not self.test_mode:
                    segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                else:
                    segment_indices = self._get_test_indices(record)
            return self.get_from_gulp(record, segment_indices)
        else:
            if not self.sparse_sampling:
                if not self.test_mode:
                    segment_indices = self._sample_indices_video(record) if self.random_shift else self._get_val_indices_video(record)
                else:
                    segment_indices = self._get_test_indices_video(record)
            else:
                if not self.test_mode:
                    segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                else:
                    segment_indices = self._get_test_indices(record)
            if self.random_shuffling:
                segment_indices = np.random.permutation(segment_indices)
            return self.get(record, segment_indices)

    def get(self, record, indices):

        if self.num_clips > 1:
            process_data_final = []
            for k in range(self.num_clips):
                images = list()
                for seg_ind in indices[k]:
                    p = int(seg_ind)
                    if self.multilabel:
                        p = p + record.start_frame
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

                process_data, label = self.transform((images, record.label))
                process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)#
            if self.multilabel:
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
            return process_data_final, label

        else:
            images = list()
            if self.multilabel:
                indices = indices + record.start_frame
            for seg_ind in indices:
                p = int(seg_ind)
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
            process_data, label = self.transform((images, record.label))
            if self.multilabel:
                # print('multilabel')
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
            return process_data, label
            
    def get_from_hdf5(self, record, indices):


        if self.num_clips > 1:
            process_data_final = []
            hdf5_video_key = record.path
            if "something" in self.root_path.lower():
                single_h5 = os.path.join(
                    self.root_path, "Something-Something-v2-frames.h5"
                )
                if os.path.isfile(single_h5):
                    if self.h5_file is None:
                        self.h5_file = h5py.File(single_h5, "r")
                    video_binary = self.h5_file[hdf5_video_key]
                else:
                    video_binary = h5py.File(os.path.join(self.root_path, "seq_h5_30fps", record.path+".h5"))[
                        hdf5_video_key
                    ]
            else:
                single_h5 = os.path.join(self.root_path, self.mode, record.path + ".h5")
                video_binary = h5py.File(single_h5, "r")[record.path.split("/")[-1]]
            for k in range(self.num_clips):
                images = list()
                for seg_ind in indices[k]:
                    p = int(seg_ind)
                    if self.multilabel:
                        p = p + record.start_frame
                    try:
                        seg_imgs = [Image.open(io.BytesIO(video_binary[p])).convert("RGB")]
                    except:
                        seg_imgs = [Image.open(io.BytesIO(video_binary[p-1])).convert("RGB")]
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

                process_data, label = self.transform((images, record.label))
                process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)#
            if self.multilabel:
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
            return process_data_final, label

        else:
            images = list()
            if self.multilabel:
                indices = indices + record.start_frame
            hdf5_video_key = record.path
            if "something" in self.root_path.lower():
                single_h5 = os.path.join(
                    self.root_path, "Something-Something-v2-frames.h5"
                )
                if os.path.isfile(single_h5):
                    if self.h5_file is None:
                        self.h5_file = h5py.File(single_h5, "r")
                    video_binary = self.h5_file[hdf5_video_key]
                else:
                    video_binary = h5py.File(os.path.join(self.root_path, "seq_h5_30fps", record.path+".h5"))[
                        hdf5_video_key
                    ]
            else:
                single_h5 = os.path.join(self.root_path, self.mode, record.path + ".h5")
                video_binary = h5py.File(single_h5, "r")[record.path.split("/")[-1]]
            indices = self._sample_indices(None, len(video_binary)-1) if self.random_shift else self._get_val_indices(None, len(video_binary)-1)
            for seg_ind in indices:
                p = int(seg_ind)
                try:
                    seg_imgs = [Image.open(io.BytesIO(video_binary[p])).convert("RGB")]
                except:
                    print(record.path, p, len(video_binary))
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
            process_data, label = self.transform((images, record.label))
            if self.multilabel:
                # print('multilabel')
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
            return process_data, label
            
    def get_from_gulp(self, record, indices):


        if self.num_clips > 1:
            process_data_final = []
            video_id = record.path
            video_data = self.gulp[video_id][0]
            for k in range(self.num_clips):
                images = list()
                for seg_ind in indices[k]:
                    p = int(seg_ind)
                    if self.multilabel:
                        p = p + record.start_frame
                    seg_imgs = Image.fromarray(video_data[p])
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

                process_data, label = self.transform((images, record.label))
                process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)#
            if self.multilabel:
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
            return process_data_final, label

        else:
            images = list()
            video_id = record.path
            video_data = self.gulp[video_id][0]
            
            for seg_ind in indices:
                p = int(seg_ind)
                try:
                    seg_imgs = [Image.fromarray(video_data[p])]
                except:
                    print(record.path, p, len(video_data))
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
            # print(len(images), images[0].shape, type(images[0]))
            process_data, label = self.transform((images, record.label))
            if self.multilabel:
                # print('multilabel')
                label = {'action_label': record.label,
                         'verb_label': record.label_verb,
                         'noun_label': record.label_noun}
            return process_data, label

    def get_video(self, record, indices):
        # print(indices)
        if self.num_clips > 1:
            process_data_final = []
            # try:
            with open(os.path.join(self.root_path, record.path), 'rb') as f:
                vr = VideoReader(f)
                for k in range(self.num_clips):
                    images = vr.get_batch(indices[k]).asnumpy()
                    images = [Image.fromarray(images[i]).convert('RGB') for i in range(self.num_segments)]

                    process_data, label = self.transform((images, record.label))
                    process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)#
            return process_data_final, label
            # except:
            #     print('Error loading {}'.format(os.path.join(self.root_path, record.path)))

        else:
            # try:
            with open(os.path.join(self.root_path, record.path), 'rb') as f:
                vr = VideoReader(f)
                images = vr.get_batch(indices).asnumpy()
                # print(images.shape)
            # print(indices)
            images = [Image.fromarray(images[i]).convert('RGB') for i in range(self.num_segments)]
            # print(len(images), images[0].size)
            process_data, label = self.transform((images, record.label))
            # print('read success', process_data.size())
            return process_data, label
            # except:
            #     print('Error loading {}'.format(os.path.join(self.root_path, record.path)))

    def __len__(self):
        return len(self.video_list)
