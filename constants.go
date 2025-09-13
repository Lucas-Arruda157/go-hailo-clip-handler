package go_hailo_clip_handler

import (
	"time"
)

const (
	// EmbeddingsJSONFilename is the filename for the embeddings
	EmbeddingsJSONFilename = "embeddings.json"

	// EmbeddingsJSONPositiveKeyName is the key name for positive embeddings in the JSON
	EmbeddingsJSONPositiveKeyName = "positive"

	// EmbeddingsJSONNegativeKeyName is the key name for negative embeddings in the JSON
	EmbeddingsJSONNegativeKeyName = "negative"

	// HailoClipApplicationInitializedMessage is the message printed on stdout when the Hailo CLIP application is initialized
	HailoClipApplicationInitializedMessage = "Calling the Hailo CLIP application..."

	// NoClassification is the message printed on stdout when there is no classification
	NoClassification = "None"

	// GenerateEmbeddingsStartMessage is the message logged when generating embeddings starts
	GenerateEmbeddingsStartMessage = "Generating CLIP embeddings..."

	// GenerateEmbeddingsCompletedMessage is the message logged when generating embeddings is completed
	GenerateEmbeddingsCompletedMessage = "CLIP embeddings generated successfully"

	// GenerateEmbeddingsJSONPathArgument is the argument for the JSON path in the generate embeddings script
	GenerateEmbeddingsJSONPathArgument = "--json-path"

	// GenerateEmbeddingsThresholdArgument is the argument for the threshold in the generate embeddings script
	GenerateEmbeddingsThresholdArgument = "--threshold"

	// HandlerStartedMessage is the message logged when the handler starts
	HandlerStartedMessage = "CLIP handler started"

	// CloseTimeout is the timeout for closing the handler
	CloseTimeout = 5 * time.Second
)

var (
	// InitialSizeBuffer is the initial size of the buffer for reading lines
	InitialSizeBuffer = 1024 * 1024 // 1 MB

	// MaxSizeBuffer is the maximum size of the buffer for reading lines
	MaxSizeBuffer = 1024 * 1024 * 10 // 10 MB

	// StdoutTag is the tag for standard output logs
	StdoutTag = "STDOUT"

	// StderrTag is the tag for standard error logs
	StderrTag = "STDERR"

	// IgnoreFirstStdoutMessages is the number of initial stdout messages to ignore
	IgnoreFirstStdoutMessages = 8

	// HandlerLoggerProducerTag is the logger producer tag for CLIP
	HandlerLoggerProducerTag = "CLIP_HANDLER"

	// GenerateEmbeddingsLoggerProducerTag is the logger producer tag for the generate embeddings process
	GenerateEmbeddingsLoggerProducerTag = "CLIP_GENERATE_EMBEDDINGS"
)