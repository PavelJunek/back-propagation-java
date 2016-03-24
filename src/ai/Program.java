/*
 * Copyright (C) 2016 Pavel Junek
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package ai;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

/**
 * This program demonstrates the use of neural network.
 *
 * @author Pavel Junek
 */
public class Program implements Runnable {

	private final File trainingFile;
	private final File validationFile;

	public Program(File trainingFile, File validationFile) {
		this.trainingFile = trainingFile;
		this.validationFile = validationFile;
	}

	@Override
	public void run() {
		try {
			PrintStream out = System.out;
			BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
			String answer;

			// Read the training and validation sets
			List<TrainingItem> trainingSet = readLearningSet(trainingFile, 64, 10);
			List<TrainingItem> validationSet = readLearningSet(validationFile, 64, 10);

			// Create the network
			// You can experiment with different sizes of the hidden layer (replace 65 with another number)
			Network network = new Network(64, 65, 10, 0.1);

			// Just for demonstration - calculate the error before the training has started
			out.printf("Initial error is %f\n", network.getError(validationSet));

			// Train the network
			// You can see the error after every training epoch
			int i = 0;
			do {
				++i;

				out.printf("Learning %d. epoch...\n", i);
				network.trainEpoch(trainingSet);
				out.printf("Error after %d. epoch is %f\n", i, network.getError(validationSet));

				out.println("Do you want to try next epoch? [Y/N] ");
				answer = in.readLine();
			} while (answer.startsWith("Y") || answer.startsWith("y"));

			// Test the network
			for (;;) {
				out.println("Enter file name to classify: ");

				answer = in.readLine();
				if (answer.length() == 0) {
					break;
				}

				File file = new File(answer);
				if (!file.exists()) {
					continue;
				}

				Item item = readItem(file, 64);
				int digit = network.classify(item);
				out.printf("This seems like %d\n", digit);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private List<TrainingItem> readLearningSet(File file, int inputSize, int outputSize) throws IOException {
		try (InputStream stream = new FileInputStream(file); Reader reader = new InputStreamReader(stream); BufferedReader lineReader = new BufferedReader(reader)) {
			List<TrainingItem> trainingSet = new ArrayList<>();
			String line;
			while ((line = lineReader.readLine()) != null) {
				trainingSet.add(TrainingItem.read(line, inputSize, outputSize));
			}
			return trainingSet;
		}
	}

	private Item readItem(File file, int inputSize) throws IOException {
		try (InputStream stream = new FileInputStream(file); Reader reader = new InputStreamReader(stream); BufferedReader lineReader = new BufferedReader(reader)) {
			String line;
			if ((line = lineReader.readLine()) == null) {
				throw new IOException("Invalid file format, must contain one line.");
			}
			return Item.read(line, inputSize);
		}
	}

	private static final String USAGE = "Usage: program <training file> <validation file>";

	public static void main(String[] args) {
		if (args.length != 2) {
			System.err.println(USAGE);
			return;
		}

		File trainingFile = new File(args[0]);
		if (!trainingFile.exists()) {
			System.err.println(USAGE);
			return;
		}

		File validationFile = new File(args[1]);
		if (!validationFile.exists()) {
			System.err.println(USAGE);
			return;
		}

		new Program(trainingFile, validationFile).run();
	}
}
