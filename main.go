package main

import (
	"fmt"
	"os"
	"encoding/csv"
	"math"
	"strconv"
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


func createLinearRegerssion(filename string) (*linearRegression, error) {
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


func main() {
	lr, err := createLinearRegerssion("data.csv")
	if err != nil {
        fmt.Fprintln(os.Stderr, "Error:", err)
        os.Exit(1)
    }
	lr.train()
	err1 := lr.storeThetas()
	if err1 != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
        os.Exit(1)
	}
}
