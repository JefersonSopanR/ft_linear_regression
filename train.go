package main

import (
	"fmt"
	"os"
	"encoding/csv"
	"math"
	"strconv"
	"image/color"
	
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type linearRegression struct {
	learningRate float64
	theta0 float64
	theta1 float64

	rawMileages	[]float64
	rawPrices []float64

	meanMileage float64
	stdDevMileage float64

	mileages []float64
	prices []float64

	lossAcc []float64
}


func createLinearRegression(filename string) (*linearRegression, error) {
	lr := &linearRegression{
		learningRate: 0.001,
		theta0: 0.0,
		theta1: 0.0,
	}
	if err := lr.readData(filename); err != nil {
        return nil, err
    }
	lr.standardize()

	return lr, nil
}

func (lr *linearRegression) readData(filename string) error{
	file, err := os.Open(filename)

	if (err != nil) {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader:= csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return fmt.Errorf("failed to read CSV: %w", err)
	}

	for _, record := range records[1:] {
		if len(record) < 2 {
            return fmt.Errorf("invalid record: %v", record)
        }
		mileage, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return fmt.Errorf("invalid mileage: %v", record[0])
		}
		price, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			return fmt.Errorf("invalid price: %v", record[1])
		}

		lr.rawMileages = append(lr.rawMileages, mileage)
		lr.rawPrices = append(lr.rawPrices, price)
	}
	return nil
}

func (lr *linearRegression) standardize() {
	sum := 0.0

	// get the mean for the mileages
	for _, m := range lr.rawMileages {
		sum += m
	}
	lr.meanMileage = sum / float64(len(lr.rawMileages))

	fmt.Printf("mean mileage:  %v\n", lr.meanMileage)

	// get the standar Deviation for the mileages -> by how the point are far from the mean on avarege
	variance := 0.0
	for _, m := range lr.rawMileages {
		diff := m - lr.meanMileage
		variance += diff * diff
	}
	lr.stdDevMileage = math.Sqrt(variance / float64(len(lr.rawMileages)))

	for _, m := range lr.rawMileages {
		scaled := (m - lr.meanMileage) / lr.stdDevMileage
		lr.mileages = append(lr.mileages, scaled)
	}
	lr.prices = lr.rawPrices
}

func (lr *linearRegression) calculateError() (float64, float64, float64) {
	t0error := 0.0
	t1error := 0.0
	totalLoss := 0.0

	for i := 0; i < len(lr.mileages); i++ {
		prediction := lr.theta0 + lr.theta1 * lr.mileages[i]
		err := prediction - lr.prices[i]

		t0error += err
		t1error += err * lr.mileages[i]
		totalLoss += math.Abs(err)
	}
	totalLoss /= float64(len(lr.mileages))

	return t0error, t1error, totalLoss
}

func (lr *linearRegression) train() {
	maxEpoch := 1000
	tolerance := 1e-7

	for epoch := 0; epoch < maxEpoch; epoch++ {
		t0error, t1error, loss := lr.calculateError()
		lr.theta0 -= t0error * lr.learningRate
		lr.theta1 -= t1error * lr.learningRate
		lr.lossAcc = append(lr.lossAcc, loss)

		if len(lr.lossAcc) > 1 {
			prev := lr.lossAcc[len(lr.lossAcc) - 2]
			curr := lr.lossAcc[len(lr.lossAcc) - 1]
			if math.Abs(curr - prev) < tolerance {
				fmt.Printf("Converged at epoch %d (loss: %.6f)\n", epoch, loss)
				break
			}
		}

		if epoch%(maxEpoch/20) == 0 {
			fmt.Printf("Epoch %d\tLoss: %.7f\n", epoch, loss)
		}
	}

	// unscale thetas 
	lr.theta1 /= lr.stdDevMileage
	lr.theta0 -= lr.theta1 * lr.meanMileage
}

func (lr *linearRegression) storeThetas() error {
	file, err := os.Create("thetas")
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	fmt.Fprintf(file, "%f,%f", lr.theta0, lr.theta1)
	return nil
}

// BONUS: Calculate precision metrics
func (lr *linearRegression) calculatePrecision() {
	fmt.Println("\n=== Model Precision Metrics ===")
	
	// Calculate mean price
	meanPrice := 0.0
	for _, p := range lr.rawPrices {
		meanPrice += p
	}
	meanPrice /= float64(len(lr.rawPrices))
	
	// Calculate metrics
	ssRes := 0.0  // residual sum of squares
	ssTot := 0.0  // total sum of squares
	mae := 0.0    // mean absolute error
	mse := 0.0    // mean squared error
	
	for i := range lr.rawMileages {
		prediction := lr.theta0 + lr.theta1 * lr.rawMileages[i]
		residual := lr.rawPrices[i] - prediction
		
		ssRes += residual * residual
		ssTot += math.Pow(lr.rawPrices[i] - meanPrice, 2)
		mae += math.Abs(residual)
		mse += residual * residual
	}
	
	n := float64(len(lr.rawPrices))
	mae /= n
	mse /= n
	rmse := math.Sqrt(mse)
	r2 := 1 - (ssRes / ssTot)
	
	fmt.Printf("R² Score (Coefficient of Determination): %.4f\n", r2)
	fmt.Printf("  → Explains %.2f%% of variance in the data\n", r2*100)
	fmt.Printf("Mean Absolute Error (MAE): %.2f\n", mae)
	fmt.Printf("Root Mean Squared Error (RMSE): %.2f\n", rmse)
	fmt.Printf("Mean Price: %.2f\n", meanPrice)
	
	// Interpretation
	if r2 > 0.9 {
		fmt.Println("→ Excellent fit!")
	} else if r2 > 0.7 {
		fmt.Println("→ Good fit")
	} else if r2 > 0.5 {
		fmt.Println("→ Moderate fit")
	} else {
		fmt.Println("→ Poor fit - consider different features")
	}
}

// BONUS: Plot just the data points
func (lr *linearRegression) plotDataPoints() error {
	p := plot.New()
	
	p.Title.Text = "Car Price vs Mileage (Data Distribution)"
	p.X.Label.Text = "Mileage (km)"
	p.Y.Label.Text = "Price"
	
	// Create scatter plot points
	pts := make(plotter.XYs, len(lr.rawMileages))
	for i := range pts {
		pts[i].X = lr.rawMileages[i]
		pts[i].Y = lr.rawPrices[i]
	}
	
	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		return fmt.Errorf("failed to create scatter plot: %w", err)
	}
	scatter.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
	scatter.GlyphStyle.Radius = vg.Points(4)
	
	p.Add(scatter)
	
	// Save to file
	if err := p.Save(8*vg.Inch, 6*vg.Inch, "data_distribution.png"); err != nil {
		return fmt.Errorf("failed to save plot: %w", err)
	}
	
	fmt.Println("✓ Data distribution plot saved to 'data_distribution.png'")
	return nil
}

// BONUS: Plot data points with regression line
func (lr *linearRegression) plotRegression() error {
	p := plot.New()
	
	p.Title.Text = "Linear Regression: Car Price vs Mileage"
	p.X.Label.Text = "Mileage (km)"
	p.Y.Label.Text = "Price"
	
	// Create scatter plot points
	pts := make(plotter.XYs, len(lr.rawMileages))
	for i := range pts {
		pts[i].X = lr.rawMileages[i]
		pts[i].Y = lr.rawPrices[i]
	}
	
	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		return fmt.Errorf("failed to create scatter plot: %w", err)
	}
	scatter.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
	scatter.GlyphStyle.Radius = vg.Points(3)
	
	// Create regression line
	minMileage := lr.rawMileages[0]
	maxMileage := lr.rawMileages[0]
	for _, m := range lr.rawMileages {
		if m < minMileage {
			minMileage = m
		}
		if m > maxMileage {
			maxMileage = m
		}
	}
	
	linePoints := plotter.XYs{
		{X: minMileage, Y: lr.theta0 + lr.theta1*minMileage},
		{X: maxMileage, Y: lr.theta0 + lr.theta1*maxMileage},
	}
	
	line, err := plotter.NewLine(linePoints)
	if err != nil {
		return fmt.Errorf("failed to create line plot: %w", err)
	}
	line.LineStyle.Width = vg.Points(2)
	line.LineStyle.Color = color.RGBA{R: 255, A: 255}
	
	// Add plots to the canvas
	p.Add(scatter, line)
	p.Legend.Add("Data points", scatter)
	p.Legend.Add("Regression line", line)
	
	// Save to file
	if err := p.Save(8*vg.Inch, 6*vg.Inch, "regression_with_line.png"); err != nil {
		return fmt.Errorf("failed to save plot: %w", err)
	}
	
	fmt.Println("✓ Regression plot saved to 'regression_with_line.png'")
	return nil
}


func main() {
	lr, err := createLinearRegression("data.csv")
	if err != nil {
        fmt.Fprintln(os.Stderr, "Error:", err)
        os.Exit(1)
    }
	
	lr.train()
	
	err1 := lr.storeThetas()
	if err1 != nil {
		fmt.Fprintln(os.Stderr, "Error:", err1)
        os.Exit(1)
	}
	
	// BONUS: Calculate and display precision
	lr.calculatePrecision()
	
	// BONUS: Plot 1 - Just the data points
	fmt.Println("\n=== Generating Plots ===")
	if err := lr.plotDataPoints(); err != nil {
		fmt.Fprintln(os.Stderr, "Data plot error:", err)
	}
	
	// BONUS: Plot 2 - Data points with regression line
	if err := lr.plotRegression(); err != nil {
		fmt.Fprintln(os.Stderr, "Regression plot error:", err)
	}
}