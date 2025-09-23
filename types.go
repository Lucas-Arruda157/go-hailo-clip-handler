package go_hailo_clip_handler

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"

	goconcurrentlogger "github.com/ralvarezdev/go-concurrent-logger"
	gostringsconvert "github.com/ralvarezdev/go-strings/convert"
)

type (
	// Classification is the struct for a Hailo CLIP classification
	Classification struct {
		Label      string
		Confidence float32
	}

	// DefaultHandler is the handler for the Hailo CLIP application
	DefaultHandler struct {
		handlerMutex                     sync.Mutex
		classificationMutex              sync.RWMutex
		isRunning                        atomic.Bool
		generateCLIPEmbeddingsPath       string
		runCLIPPath                      string
		positiveLabels                   []string
		negativeLabels                   []string
		classification                   *Classification
		stdoutLinesRead                  int
		clipApplicationInitialized       atomic.Bool
		clipApplicationStarted           atomic.Bool
		logger                           goconcurrentlogger.Logger
		handlerLoggerProducer            goconcurrentlogger.LoggerProducer
		generateEmbeddingsLoggerProducer goconcurrentlogger.LoggerProducer
		embeddingsJSONPath               string
		minimumConfidenceThreshold       float32
		debug                            bool
		hasStartedSending             atomic.Bool
		classificationsChSize 		   int	
		classificationsCh             chan *Classification
	}
)

// NewClassification creates a new Classification instance.
//
// Parameters:
//
// label: The label for the classification.
// confidence: The confidence score for the classification.
//
// Returns:
//
// A pointer to a Classification instance or an error if any parameter is invalid.
func NewClassification(
	label string,
	confidence float32,
) (*Classification, error) {
	// Check if the confidence is within the valid range [0.0, 1.0]
	if confidence < 0.0 || confidence > 1.0 {
		return nil, fmt.Errorf(
			"confidence must be in [0.0, 1.0], got %f",
			confidence,
		)
	}

	// Create a new Classification instance
	classification := &Classification{
		Label:      label,
		Confidence: confidence,
	}

	return classification, nil
}

// NewClassificationFromString creates a new Classification instance from a string.
//
// Parameters:
//
// s: The string representation of the classification in the format "label confidence".
//
// Returns:
//
// A pointer to a Classification instance or an error if the string is invalid.
func NewClassificationFromString(s string) (*Classification, error) {
	// Split the string into fields
	fields := strings.Fields(s)
	label := strings.Join(fields[:len(fields)-1], " ")
	confidenceStr := fields[len(fields)-1]

	// Parse the confidence
	var confidence float32
	if err := gostringsconvert.ToFloat32(
		confidenceStr,
		&confidence,
	); err != nil {
		return nil, fmt.Errorf("failed to parse confidence: %w", err)
	}

	// Create a new Classification instance
	return NewClassification(label, confidence)
}

// GetLabel returns the label of the classification.
//
// Returns:
//
// The label of the classification.
func (c *Classification) GetLabel() string {
	return c.Label
}

// GetConfidence returns the confidence score of the classification.
//
// Returns:
//
// The confidence score of the classification.
func (c *Classification) GetConfidence() float32 {
	return c.Confidence
}

// NewDefaultHandler creates a new DefaultHandler instance.
//
// Parameters:
//
// generateCLIPEmbeddingsPath: Path to the .sh file that generates CLIP embeddings.
// embeddingsJSONPath: Path to the embeddings JSON file.
// runCLIPPath: Path to the .sh file that runs CLIP.
// positiveLabels: Slice of positive labels for classification.
// negativeLabels: Slice of negative labels for classification (optional, can be nil).
// minimumConfidenceThreshold: Minimum confidence threshold for valid classifications.
// logger: Logger instance for logging messages.
// classificationsChSize: Size of the classifications channel (must be greater than 0).
// debug: Whether to enable debug logging.
//
// Returns:
//
// A pointer to a DefaultHandler instance or an error if any parameter is invalid.
func NewDefaultHandler(
	generateCLIPEmbeddingsPath,
	embeddingsJSONPath string,
	runCLIPPath string,
	positiveLabels []string,
	negativeLabels []string,
	minimumConfidenceThreshold float32,
	logger goconcurrentlogger.Logger,
	classificationsChSize int,
	debug bool,
) (*DefaultHandler, error) {
	// Check if the logger is nil
	if logger == nil {
		return nil, goconcurrentlogger.ErrNilLogger
	}

	// Check if the runCLIPPath is empty
	if runCLIPPath == "" {
		return nil, ErrEmptyRunCLIPPath
	}

	// Check if the positiveLabels is nil
	if positiveLabels == nil {
		return nil, ErrNilPositiveLabels
	}

	// Check if the positiveLabels is empty
	if len(positiveLabels) == 0 {
		return nil, ErrEmptyPositiveLabels
	}

	// Check if any positive label is empty
	for _, label := range positiveLabels {
		if strings.TrimSpace(label) == "" {
			return nil, ErrEmptyPositiveLabel
		}
	}

	// Check if any negative label is empty
	for _, label := range negativeLabels {
		if strings.TrimSpace(label) == "" {
			return nil, ErrEmptyNegativeLabel
		}
	}

	// Check if the minimumConfidenceThreshold is valid
	if minimumConfidenceThreshold < 0.0 || minimumConfidenceThreshold > 1.0 {
		return nil, ErrInvalidMinimumConfidenceThreshold
	}

	// Check if the classificationsChSize is valid
	if classificationsChSize <= 0 {
		return nil, ErrInvalidClassificationsChSize
	}

	// Create a new DefaultHandler instance
	return &DefaultHandler{
		generateCLIPEmbeddingsPath: generateCLIPEmbeddingsPath,
		embeddingsJSONPath:         embeddingsJSONPath,
		runCLIPPath:                runCLIPPath,
		positiveLabels:             positiveLabels,
		negativeLabels:             negativeLabels,
		minimumConfidenceThreshold: minimumConfidenceThreshold,
		logger:                     logger,
		debug:                      debug,
		classificationsChSize:      classificationsChSize,
	}, nil
}

// generateEmbeddingsJSONContent generates the content for the embeddings JSON file.
//
// Returns:
//
// A string containing the JSON content.
func (h *DefaultHandler) generateEmbeddingsJSONContent() string {
	var builder strings.Builder

	// Add the opening brace and positive labels key
	builder.WriteString("{\n")
	builder.WriteString("\t\"")
	builder.WriteString(EmbeddingsJSONPositiveKeyName)
	builder.WriteString("\": [\n")

	// Add the positive labels
	for i, label := range h.positiveLabels {
		builder.WriteString("\t\t\"")
		builder.WriteString(label)
		builder.WriteString("\"")
		if i < len(h.positiveLabels)-1 {
			builder.WriteString(",\n")
		} else {
			builder.WriteString("\n")
		}
	}

	// Add the negative labels
	if h.negativeLabels == nil {
		builder.WriteString("\t]\n")
	} else {
		builder.WriteString("\t],\n")
		builder.WriteString("\t\"")
		builder.WriteString(EmbeddingsJSONNegativeKeyName)
		builder.WriteString("\": [\n")
		for i, label := range h.negativeLabels {
			builder.WriteString("\t\t\"")
			builder.WriteString(label)
			builder.WriteString("\"")
			if i < len(h.negativeLabels)-1 {
				builder.WriteString(",\n")
			} else {
				builder.WriteString("\n")
			}
		}
		builder.WriteString("\t]\n")
	}

	// Add the closing brace
	builder.WriteString("}\n")
	return builder.String()
}

// GenerateEmbeddings generates CLIP embeddings by executing the specified script.
//
// Parameters:
//
// ctx: Context for managing cancellation and timeouts.
//
// Returns:
//
// An error if any issue occurs during the execution of the script.
func (h *DefaultHandler) GenerateEmbeddings(ctx context.Context) error {
	// Check if the generateCLIPEmbeddingsPath is empty
	if h.generateCLIPEmbeddingsPath == "" {
		return ErrEmptyGenerateEmbeddingsPath
	}

	// Create a logger producer
	generateEmbeddingsLoggerProducer, err := h.logger.NewProducer(
		GenerateEmbeddingsLoggerProducerTag,
		h.debug,
	)
	if err != nil {
		return fmt.Errorf(
			"failed to create generate embeddings logger producer: %w",
			err,
		)
	}
	h.generateEmbeddingsLoggerProducer = generateEmbeddingsLoggerProducer
	defer h.generateEmbeddingsLoggerProducer.Close()

	// Log the start of generating embeddings
	h.generateEmbeddingsLoggerProducer.Info(GenerateEmbeddingsStartMessage)

	// Generate JSON file
	jsonContent := h.generateEmbeddingsJSONContent()

	// Ensure the directory for the JSON file exists
	parentDir := filepath.Dir(h.embeddingsJSONPath)
	if err := os.MkdirAll(parentDir, os.ModePerm); err != nil {
		return fmt.Errorf(
			"failed to create directory for embeddings JSON file: %w",
			err,
		)
	}

	// Write the JSON content to the file
	if err := os.WriteFile(
		h.embeddingsJSONPath,
		[]byte(jsonContent),
		0644,
	); err != nil {
		return fmt.Errorf("failed to write embeddings JSON file: %w", err)
	}

	// Check if the generate embeddings executable exists
	if _, err := os.Stat(h.generateCLIPEmbeddingsPath); errors.Is(
		err,
		os.ErrNotExist,
	) {
		return fmt.Errorf(
			"generate embeddings executable not found at path: %s",
			h.generateCLIPEmbeddingsPath,
		)
	}

	// Arguments (do not include the executable itself)
	args := []string{
		GenerateEmbeddingsJSONPathArgument,
		h.embeddingsJSONPath,
		GenerateEmbeddingsThresholdArgument,
		fmt.Sprintf("%f", h.minimumConfidenceThreshold),
	}

	// Execute the command
	cmd := exec.CommandContext(ctx, h.generateCLIPEmbeddingsPath, args...)

	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	// Run and wait
	if err := cmd.Run(); err != nil {
		h.generateEmbeddingsLoggerProducer.Warning(
			fmt.Sprintf(
				"Embeddings script failed: %v; stderr: %s",
				err,
				strings.TrimSpace(stderrBuf.String()),
			),
		)
		return fmt.Errorf("generate embeddings script error: %w", err)
	}

	// Log outputs if debug
	if h.generateEmbeddingsLoggerProducer.IsDebug() {
		stdout := strings.TrimSpace(stdoutBuf.String())
		if stdout != "" {
			h.generateEmbeddingsLoggerProducer.Debug(
				fmt.Sprintf(
					"embeddings script stdout: %s",
					stdout,
				),
			)
		}
		stderr := strings.TrimSpace(stderrBuf.String())
		if stderr != "" {
			h.generateEmbeddingsLoggerProducer.Debug(
				fmt.Sprintf(
					"embeddings script stderr: %s",
					stderr,
				),
			)
		}
	}

	h.generateEmbeddingsLoggerProducer.Info(GenerateEmbeddingsCompletedMessage)
	return nil
}

// IsRunning returns whether the handler is currently running.
//
// Returns:
//
// True if the handler is running, false otherwise.
func (h *DefaultHandler) IsRunning() bool {
	return h.isRunning.Load()
}

// runToWrap is the internal function to read incoming classifications from the CLIP application.
//
// Parameters:
//
// ctx: Context for managing cancellation and timeouts.
// cancelFn: Function to call to cancel the context.
//
// Returns:
//
// An error if any issue occurs during execution.
func (h *DefaultHandler) runToWrap(ctx context.Context, cancelFn context.CancelFunc) error {
	// Reset the stdout lines read counter
	h.stdoutLinesRead = 0

	// Log the start of reading measures
	h.handlerLoggerProducer.Info(HandlerStartedMessage)

	// Check if the run clip executable exists
	if _, err := os.Stat(h.runCLIPPath); errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf(
			"run clip executable not found at path: %s",
			h.runCLIPPath,
		)
	}

	// Execute the command with a context
	cmd := exec.CommandContext(ctx, h.runCLIPPath)

	// Stream output line by line
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe error: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe error: %w", err)
	}

	// Start the command
	if err = cmd.Start(); err != nil {
		return fmt.Errorf("start command error: %w", err)
	}

	// Create an error group to wait for all goroutines to finish
	g := &errgroup.Group{}

	// Stream stdout
	g.Go(
		goconcurrentlogger.CancelContextAndLogOnError(
			ctx,
			cancelFn,
			func(ctx context.Context) error {
				return h.scanLines(
					ctx,
					StdoutTag,
					stdout,
					h.handleStdoutLine,
				)
			},
			h.handlerLoggerProducer,
		),
	)

	// Stream stderr
	g.Go(
		goconcurrentlogger.CancelContextAndLogOnError(
			ctx,
			cancelFn,
			func(ctx context.Context) error {
				return h.scanLines(
					ctx,
					StderrTag,
					stderr,
					h.handleStderrLine,
				)
			},
			h.handlerLoggerProducer,
		),
	)

	// Wait for completion or context cancel
	if err = g.Wait(); err != nil && !errors.Is(err, context.Canceled) {
		return fmt.Errorf("error reading lines: %w", err)
	}

	// Log the process exit

	h.handlerLoggerProducer.Info("CLIP process exiting...")

	// Close the stdout and stderr pipes
	_ = stdout.Close()
	_ = stderr.Close()

	// Signal the process to stop (SIGINT)
	_ = cmd.Process.Signal(os.Interrupt)

	// Wait for the process to exit or timeout
	done := make(chan struct{})
	go func() {
		cmd.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Process exited gracefully
		h.handlerLoggerProducer.Info("CLIP process exited gracefully")
	case <-time.After(CloseTimeout):
		// Timeout, force kill
		_ = cmd.Process.Kill()
		h.handlerLoggerProducer.Warning("CLIP process killed after timeout")
	}
	return nil
}

// Run reads incoming classifications from the CLIP application.
//
// Parameters:
//
// ctx: Context for managing cancellation and timeouts.
// cancelFn: Function to call to cancel the context.
//
// Returns:
//
// An error if any issue occurs during reading or processing classifications.
func (h *DefaultHandler) Run(ctx context.Context, cancelFn context.CancelFunc) error {
	h.handlerMutex.Lock()

	// Check if it's already running
	if h.IsRunning() {
		h.handlerMutex.Unlock()
		return ErrHandlerAlreadyRunning
	}
	defer h.close()

	// Set running to true
	h.isRunning.Store(true)

	// Reset clip application state
	h.clipApplicationInitialized.Store(false)
	h.clipApplicationStarted.Store(false)

	// Reset current classification
	h.classification = nil

	// Create the classifications channel
	h.classificationsCh = make(chan *Classification, h.classificationsChSize)

	h.handlerMutex.Unlock()

	// Create a logger producer
	handlerLoggerProducer, err := h.logger.NewProducer(
		HandlerLoggerProducerTag,
		h.debug,
	)
	if err != nil {
		return fmt.Errorf("failed to create handler logger producer: %w", err)
	}
	h.handlerLoggerProducer = handlerLoggerProducer
	defer h.handlerLoggerProducer.Close()

	return goconcurrentlogger.CancelContextAndLogOnError(
		ctx,
		cancelFn,
		func(ctx context.Context) error {
			return h.runToWrap(ctx, cancelFn)
		},
		h.handlerLoggerProducer,
	)()
}

// close closes the handler and releases resources, but the context must be cancelled externally.
func (h *DefaultHandler) close() {
	h.handlerMutex.Lock()

	// Check if the handler is already closed
	if !h.IsRunning() { 
		h.handlerMutex.Unlock()
		return
	}

	// Mark the handler as closed
	h.isRunning.Store(false)

	h.handlerMutex.Unlock()

	// Reset has started sending state
	h.hasStartedSending.Store(false)

	// Close the classifications channel
	close(h.classificationsCh)
}

// StartSendingClassifications sets the handler to start sending classifications through the classifications channel.
//
// Returns:
//
// An error if the handler is not running.
func (h *DefaultHandler) StartSendingClassifications() error {
	h.handlerMutex.Lock()
	defer h.handlerMutex.Unlock()
	if !h.IsRunning() {
		return ErrHandlerIsNotRunning
	}
	h.hasStartedSending.Store(true)
	return nil
}

// StopSendingClassifications sets the handler to stop sending classifications through the classifications channel.
//
// Returns:
//
// An error if the handler is not running.
func (h *DefaultHandler) StopSendingClassifications() error {
	h.handlerMutex.Lock()
	defer h.handlerMutex.Unlock()
	if !h.IsRunning() {
		return ErrHandlerIsNotRunning
	}
	h.hasStartedSending.Store(false)
	return nil
}

// GetClassificationsChannel returns the channel through which classifications are sent.
//
// Returns:
//
// A read-only channel of classifications, or an error if the handler is not running.
func (h *DefaultHandler) GetClassificationsChannel() (<-chan *Classification, error) {
	h.handlerMutex.Lock()
	defer h.handlerMutex.Unlock()
	if !h.IsRunning() {
		return nil, ErrHandlerIsNotRunning
	}
	return h.classificationsCh, nil
}


// scanLines reads lines from the provided reader and processes them using the given lineHandler.
//
// Parameters:
//
// ctx: Context for managing cancellation and timeouts.
// tag: Tag to identify the source of the lines (e.g., "stdout" or "stderr").
// r: Reader to read lines from.
// lineHandler: Function to process each line.
//
// Returns:
//
// An error if any issue occurs during reading or processing lines.
func (h *DefaultHandler) scanLines(
	ctx context.Context,
	tag string,
	r interface{ Read([]byte) (int, error) },
	lineHandler func(string) error,
) error {
	// Check if the lineHandler is nil
	if lineHandler == nil {
		return ErrNilLineHandler
	}

	// Create a new scanner
	sc := bufio.NewScanner(r)

	// Set the buffer size
	buf := make([]byte, 0, InitialSizeBuffer)
	sc.Buffer(buf, MaxSizeBuffer)

	for sc.Scan() {
		select {
		case <-ctx.Done():
			h.handlerLoggerProducer.Info(
				fmt.Sprintf(
					"Context done while reading lines from %s: %v",
					tag,
					ctx.Err(),
				),
			)
			// Return context error
			return ctx.Err()
		default:
			// Read the line
			line := strings.TrimSpace(sc.Text())

			// Process the line
			if h.handlerLoggerProducer.IsDebug() {
				h.handlerLoggerProducer.Debug(
					fmt.Sprintf(
						"Received line from %s: %s",
						tag,
						line,
					),
				)
			}

			// Handle the line
			if err := lineHandler(line); err != nil {
				return err
			}
		}
	}

	// Check for scanning errors
	if err := sc.Err(); err != nil {
		return fmt.Errorf("scan error: %w", err)
	}
	return nil
}

// handleStdoutLine processes a single line from stdout.
//
// Parameters:
//
// line: The line to process.
//
// Returns:
//
// An error if any issue occurs during processing the line.
func (h *DefaultHandler) handleStdoutLine(line string) error {
	// Increment the stdout lines read counter
	h.stdoutLinesRead++

	// Check if the message should be ignored
	if h.stdoutLinesRead <= IgnoreFirstStdoutMessages {
		return nil
	}

	// Check if the CLIP Application has been initialized
	if !h.clipApplicationInitialized.Load() {
		if line == HailoCLIPApplicationInitialized {
			h.clipApplicationInitialized.Store(true)
			h.handlerLoggerProducer.Info(HailoCLIPApplicationInitializedMessage)
		}
		return nil
	}

	// Check if the CLIP Application has started
	if !h.clipApplicationStarted.Load() {
		if line == HailoCLIPApplicationStarted {
			h.clipApplicationStarted.Store(true)
			h.handlerLoggerProducer.Info(HailoCLIPApplicationStartedMessage)
		}
		return nil
	}

	// Lock the measures map for writing
	h.classificationMutex.Lock()
	defer h.classificationMutex.Unlock()

	// Check if there is a classification in the line
	if line == NoClassification {
		if h.handlerLoggerProducer.IsDebug() {
			h.handlerLoggerProducer.Debug("No classification detected")
		}
		h.classification = nil

		// Send nil classification if started sending
		if h.hasStartedSending.Load() {
			select {
			case h.classificationsCh <- nil:
				if h.handlerLoggerProducer.IsDebug() {
					h.handlerLoggerProducer.Debug(
						"Sent no classification message",
					)
				}
			default:
				if h.handlerLoggerProducer.IsDebug() {
					h.handlerLoggerProducer.Debug(
						"Classifications channel is full, dropping no classification message",
					)
				}
			}
		}	
		return nil
	}

	// Create a classification from the given string
	classification, err := NewClassificationFromString(line)
	if err != nil {
		h.handlerLoggerProducer.Warning(
			fmt.Sprintf(
				"Failed to parse classification: %v",
				err,
			),
		)
		return nil // Ignore parsing errors
	}

	// Check if the confidence is below the threshold
	if classification.GetConfidence() < h.minimumConfidenceThreshold {
		if h.handlerLoggerProducer.IsDebug() {
			h.handlerLoggerProducer.Debug(
				fmt.Sprintf(
					"Ignoring classification of label '%s' with low confidence: %f",
					classification.GetLabel(),
					classification.GetConfidence(),
				),
			)
		}
		return nil
	}

	// Update the current classification
	if h.classification != classification {
		if h.handlerLoggerProducer.IsDebug() {
			h.handlerLoggerProducer.Debug(
				fmt.Sprintf(
					"New classification detected for label '%s' with confidence: %f",
					classification.GetLabel(),
					classification.GetConfidence(),
				),
			)
		}
	}
	h.classification = classification

	// Send the classification if started sending
	if h.hasStartedSending.Load() {
		select {
		case h.classificationsCh <- classification:
			if h.handlerLoggerProducer.IsDebug() {
				h.handlerLoggerProducer.Debug(
					fmt.Sprintf(
						"Sent classification: label='%s', confidence=%f",
						classification.GetLabel(),
						classification.GetConfidence(),
					),
				)
			}
		default:
			if h.handlerLoggerProducer.IsDebug() {
				h.handlerLoggerProducer.Debug(
					"Classifications channel is full, dropping classification message",
				)
			}
		}
	}
	return nil
}

// GetClassification returns a copy of the current classification.
//
// Returns:
//
// A copy of the current classification or nil if there is no classification, along with an error if any issue occurs.
func (h *DefaultHandler) GetClassification() (*Classification, error) {
	// Lock the classification for reading
	h.classificationMutex.RLock()
	defer h.classificationMutex.RUnlock()

	// If there is no classification, return nil
	if h.classification == nil {
		return nil, nil
	}

	// Create a copy of the classification
	classificationCopy := *h.classification
	return &classificationCopy, nil
}

// handleStderrLine processes a single line from stderr.
//
// Parameters:
//
// line: The line to process.
//
// Returns:
//
// An error if any issue occurs during processing the line.
func (h *DefaultHandler) handleStderrLine(line string) error {
	if h.clipApplicationStarted.Load() {
		// Log the stderr line as a warning
		h.handlerLoggerProducer.Warning(fmt.Sprintf("stderr: %s", line))
	}
	return nil
}
