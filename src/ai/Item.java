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

/**
 * This class represents one line in the input data set.
 *
 * @author Pavel Junek
 */
public class Item {

	/**
	 * The array of input values.
	 */
	public final double[] x;

	/**
	 * Initializes the new instance.
	 *
	 * @param x an array of input values
	 */
	public Item(double[] x) {
		this.x = x;
	}

	/**
	 * Creates a new item from the given array of strings.
	 *
	 * @param line a string containing the values (the line must contain only
	 * numeric values separated by commas)
	 * @param size count of values to use (the array must contain at least count
	 * numeric values)
	 * @return the item created from the given array
	 */
	public static Item read(String line, int size) {
		String[] parts = line.split(",");
		if (parts.length < size) {
			throw new IllegalArgumentException(size + " values required, " + parts.length + " given");
		}

		double[] x = new double[size];
		for (int i = 0; i < size; ++i) {
			x[i] = Double.valueOf(parts[i]);
		}

		return new Item(x);
	}
}
