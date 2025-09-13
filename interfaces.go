package go_hailo_clip_handler

import (
	"context"
)

type (
	// Classification is the interface for a classification result
	Classification interface {
		GetConfidence() float32
		GetLabel() string
	}

	// ClassificationHandler is the interface to parse inference lines from the Hailo CLIP application
	ClassificationHandler interface {
		ParseLine(line string) (Classification, error)
		CreateCopy(Classification) (Classification, error)
	}

	// Handler is the interface to handle the Hailo CLIP application
	Handler interface {
		GenerateEmbeddings() error
		Run(ctx context.Context, stopFn func()) error
		IsRunning() bool
		GetClassification() (Classification, error)
	}
)
