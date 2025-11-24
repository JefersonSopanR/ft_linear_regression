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
}


func createLinearRegerssion(filename string) *linearRegression {
	lr := &linearRegression{
		learningRate: 0.01,
		theta0: 0.0,
		theta1: 0.0,
	}

	lr.readData(filename)
	lr.standardize()

	return lr
}

func (lr *linearRegression) readData(filename string) {
	file, _ := os.Open(filename)
	defer file.Close()

	reader:= csv.NewReader(file)
	records, _ := reader.ReadAll()

	for _, record := range records[1:] {
		mileage, _ := strconv.ParseFloat(record[0], 64)
		price, _ := strconv.ParseFloat(record[1], 64)

		lr.rawMileages = append(lr.rawMileages, mileage)
		lr.rawPrices = append(lr.rawPrices, price)
	}

}

func (lr *linearRegression) standardize() {
	sum := 0.0

	// get the mean for the mileages
	for _, m := range lr.rawMileages {
		sum += m
	}
	lr.meanMileage = sum / float64(len(lr.rawMileages))

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


func main() {
	lr := createLinearRegerssion("data.csv")
	for _, rawPrice := range lr.rawPrices {
		fmt.Println(rawPrice)
	}
	t0, t1, err := lr.calculateError()
	fmt.Println(t0)
	fmt.Println(t1)
	fmt.Println(err)
}
