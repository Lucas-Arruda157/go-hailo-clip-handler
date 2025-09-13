package go_hailo_clip_handler

import (
	"context"
)

type (
	// Handler is the interface to handle the Hailo CLIP application
	Handler interface {
		GenerateEmbeddings() error
		Run(ctx context.Context, stopFn func()) error
		IsRunning() bool
		GetClassification() (*Classification, error)
	}
)
