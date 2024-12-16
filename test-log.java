package com.example.logging;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ComplexProcessor {
    private static final Logger LOGGER = Logger.getLogger(ComplexProcessor.class.getName());

    public static void main(String[] args) {
        LOGGER.info("ComplexProcessor application started.");

        List<String> data = fetchDataFromSource();
        if (data.isEmpty()) {
            LOGGER.warning("No data fetched from source.");
        } else {
            LOGGER.info("Data fetched successfully.");
        }

        try {
            processData(data);
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error processing data: ", e);
        }

        LOGGER.info("ComplexProcessor application ended.");
    }

    private static List<String> fetchDataFromSource() {
        LOGGER.fine("Entering fetchDataFromSource method.");
        List<String> data = new ArrayList<>();
        // Simulate data fetching
        try {
            // Simulate potential delay or issue
            Thread.sleep(1000);
            data.add("Sample Data 1");
            data.add("Sample Data 2");
            LOGGER.fine("Data fetched: " + data);
        } catch (InterruptedException e) {
            LOGGER.log(Level.WARNING, "Data fetching interrupted: ", e);
        }
        LOGGER.fine("Exiting fetchDataFromSource method.");
        return data;
    }

    private static void processData(List<String> data) throws Exception {
        LOGGER.fine("Entering processData method with data: " + data);
        if (data == null || data.isEmpty()) {
            LOGGER.warning("No data to process.");
            throw new Exception("Data is null or empty.");
        }
        for (String item : data) {
            LOGGER.fine("Processing item: " + item);
            // Simulate processing
            if ("Sample Data 2".equals(item)) {
                LOGGER.warning("Encountered known issue with item: " + item);
            }
        }
        LOGGER.fine("Exiting processData method.");
    }
}