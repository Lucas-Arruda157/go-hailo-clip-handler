package go_hailo_clip_handler

import (
	"context"
)

type (
	// Handler is the interface to handle the Hailo CLIP application
	Handler interface {
		GenerateEmbeddings(ctx context.Context) error
		Run(ctx context.Context, cancelFn context.CancelFunc) error
		IsRunning() bool
		GetClassification() (*Classification, error)
	}
)
