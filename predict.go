package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type model struct {
	theta0 float64
	theta1 float64
}

func loadThetas() (*model, error) {
	file, err := os.Open("thetas")
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		return nil, fmt.Errorf("empty thetas file")
	}

	line := scanner.Text()
	parts := strings.Split(line, ",")
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid thetas format")
	}

	theta0, err := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	if err != nil {
		return nil, fmt.Errorf("invalid theta0: %w", err)
	}

	theta1, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err != nil {
		return nil, fmt.Errorf("invalid theta1: %w", err)
	}

	return &model{theta0: theta0, theta1: theta1}, nil
}

func (m *model) estimatePrice(mileage float64) float64 {
	return m.theta0 + (m.theta1 * mileage)
}

func main() {
	// Load trained model
	model, err := loadThetas()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error loading model:", err)
		os.Exit(1)
	}

	fmt.Printf("Model loaded: θ0=%.6f, θ1=%.6f\n\n", model.theta0, model.theta1)

	// Prompt for mileage
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Enter the mileage of the car: ")
	
	input, err := reader.ReadString('\n')
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error reading input:", err)
		os.Exit(1)
	}

	// Parse mileage
	mileage, err := strconv.ParseFloat(strings.TrimSpace(input), 64)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid mileage. Please enter a number.")
		os.Exit(1)
	}

	if mileage < 0 {
		fmt.Fprintln(os.Stderr, "Mileage cannot be negative.")
		os.Exit(1)
	}

	// Estimate price
	price := model.estimatePrice(mileage)
	
	if price < 0 {
		fmt.Printf("Estimated price: 0 (model predicts negative value: %.2f)\n", price)
		fmt.Println("Note: This car has very high mileage beyond training data range.")
	} else {
		fmt.Printf("Estimated price for %.0f km: %.2f\n", mileage, price)
	}
}