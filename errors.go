package go_hailo_clip_handler

import (
	"errors"
)

var (
	ErrNilHandler                  = errors.New("handler cannot be nil")
	ErrNilLineHandler              = errors.New("line handler cannot be nil")
	ErrNilPositiveLabels           = errors.New("positive labels cannot be nil")
	ErrEmptyGenerateEmbeddingsPath = errors.New("generate embeddings path cannot be empty")
	ErrEmptyRunClipPath            = errors.New("run clip path cannot be empty")
	ErrHandlerAlreadyRunning       = errors.New("handler is already running")
	ErrEmptyPositiveLabels         = errors.New("positive labels cannot be empty")
	ErrEmptyPositiveLabel 		  = errors.New("positive label cannot be empty")
	ErrEmptyNegativeLabel 		  = errors.New("negative label cannot be empty")
	ErrInvalidMinimumConfidenceThreshold = errors.New("minimum confidence threshold must be between 0.0 and 1.0")
	ErrNilClassificationHandler = errors.New("classification handler cannot be nil")
)
