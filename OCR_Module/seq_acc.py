import paddle
import numpy as np

from ctc_decoder import CTCDecoder


class SeqAcc(paddle.metric.Metric):
    def __init__(self, blank=0, mnc_type=True, *args, **kwargs):
        """
        计算序列准确率类 - 可逐字计算是否匹配和推理结果中字符存在于Label的比例
        :param blank: 分隔符的index
        :param mnc_type: 默认按输入为MNC（Max len, Batch size, Classes）的情况来计算，若为False则为NMC

        Tips：
        网络部分输出的结果格式默认按MNC（Max len, Batch size, Classes）来计算

        Example:
        model = paddle.Model(OCRNet())
        model.prepare(metrics=SeqAcc(blank=0))

        or

        seq_acc = SeqAcc(blank=0)
        for mini_batch in data_loader():
            output = model(...)
            match_acc, include_acc = seq_acc(output, label)
        seq_acc.reset()


        AI Studio Example: https://aistudio.baidu.com/aistudio/projectdetail/1100507
        """
        super(SeqAcc, self).__init__()
        self.blank = blank
        self.mnc_type = mnc_type

        self.match_true_num = 0
        self.match_label_num = 0
        self.include_true_num = 0
        self.include_label_num = 0

    def _compute_match(self, seq, label):
        count = len([char_id for char_id in range(min(len(seq), len(label))) if seq[char_id] == label[char_id]])
        self.match_true_num += count
        self.match_label_num += len(label)

    def _compute_include(self, seq, label):
        seq = np.unique(seq).tolist()
        label = np.unique(label).tolist()
        max_len = max(len(seq), len(label))
        if len(seq) == max_len:
            sample = seq
            group = label
        else:
            sample = label
            group = seq
        count = len([char_id for char_id in range(max_len) if sample[char_id] in group])
        self.include_true_num += count
        self.include_label_num += len(sample)

    def update(self, seqs, labels, *args):
        """
        计算Acc
        :param seqs: Max length, Batch size, Classes
        :param labels: Batch size, Classes
        """
        if isinstance(seqs, paddle.nn.Layer):
            seqs = seqs.numpy()
        if isinstance(labels, paddle.nn.Layer):
            labels = labels.numpy()
        if self.mnc_type:
            seqs = np.transpose(seqs, [1, 0, 2])
        seqs = np.argmax(seqs, axis=-1)

        # CTC-Decoder
        for sample_id in range(seqs.shape[0]):
            seq = seqs[sample_id].tolist()
            label = labels[sample_id]
            label = label[label != self.blank].tolist()

            seq = CTCDecoder.ctc_decoder(seq, self.blank)
            self._compute_match(seq, label)
            self._compute_include(seq, label)
        return self.match_true_num / self.match_label_num * 100, self.include_true_num / self.include_label_num * 100

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.match_true_num = 0
        self.match_label_num = 0
        self.include_true_num = 0
        self.include_label_num = 0

    def accumulate(self):
        match_acc = f"{self.match_true_num / self.match_label_num * 100:.4f}%"
        include_acc = f"{self.include_true_num / self.include_label_num * 100:.4f}%"
        return [match_acc, include_acc]

    def name(self):
        """
        Return name of metric instance.
        """
        return ["Seq-Match Acc", "Seq-Include Acc"]


if __name__ == '__main__':
    from paddle.io import Dataset


    class TmpData(Dataset):
        def __init__(self):
            super(TmpData, self).__init__()

        def __getitem__(self, item):
            # Input: 2022302 -> 2232
            # Label: 2012202 -> 31222
            # Match Acc = 1(第0个字符、第3个字符匹配) / 5(Label字符总量) = 0.2
            # Include Acc = 2(1、2两个字符) / 3(1、2、3三个字符) = 0.67
            return np.array([[0.1, 0.3, 0.5, 0.1],
                             [0.5, 0., 0.1, 0.4],
                             [0.1, 0.3, 0.5, 0.1],
                             [0.1, 0.3, 0.5, 0.1],
                             [0.1, 0.3, 0.1, 0.5],
                             [0.7, 0., 0.1, 0.2],
                             [0.1, 0.3, 0.5, 0.1]]).astype("float32"), \
                   np.array(([3],
                             [1],
                             [2],
                             [2],
                             [2])).astype("int64")

        def __len__(self):
            return 1


    class TmpLayer(paddle.nn.Layer):
        def __init__(self):
            super(TmpLayer, self).__init__()

        def forward(self, inputs):
            return paddle.tensor.transpose(inputs, [1, 0, 2])


    model = paddle.Model(TmpLayer(),
                         inputs=[paddle.static.InputSpec([6, 4], name="inp")],
                         labels=[paddle.static.InputSpec([6, 1], dtype="int64", name="lab")])
    model.prepare(metrics=SeqAcc())
    model.evaluate(TmpData())
