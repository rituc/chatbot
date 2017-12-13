import tensorflow as tf
from collections import namedtuple

#model parameters
tf.flags.DEFINE_integer("vocab_size", 90000,"The size of the vocabulary.")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_context_len", 160, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 80, "Truncate utterance to this length")

#pre-trained enbeddings
tf.flags.DEFINE_string("glove_parh", None, "Path to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", None, "Path to vocabulary.txt file")

#Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 16, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "OPtimizer Name (Adam, Adagrad etc)")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
	"HParams",
	[
		"vocab_size",
		"embedding_dim",
		"rnn_dim",
		"max_context_len",
		"max_utterance_len",

		"glove_parh",
		"vocab_path",

		"learning_rate"
		"batch_size",
		"eval_batch_size",
		"optimizer"
	]
)

def create_hparams():
	return HParams(
		vocab_size=FLAGS.vocab_size,
		embedding_dim=FLAGS.embedding_dim,
		rnn_dim=FLAGS.rnn_dim,
		max_context_len=FLAGS.max_context_len,
		max_utterance_len = FLAGS.max_utterance_len,
		glove_parh = FLAGS.glove_parh,
		vocab_path=FLAGS.vocab_path,
		learning_rate=FLAGS.learning_rate,
		batch_size=FLAGS.batch_size,
		eval_batch_size=FLAGS.eval_batch_size,
		optimizer=FLAGS.optimizer
	)