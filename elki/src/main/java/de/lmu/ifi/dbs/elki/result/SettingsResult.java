package de.lmu.ifi.dbs.elki.result;

/*
 This file is part of ELKI:
 Environment for Developing KDD-Applications Supported by Index-Structures

 Copyright (C) 2015
 Ludwig-Maximilians-Universität München
 Lehr- und Forschungseinheit für Datenbanksysteme
 ELKI Development Team

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import java.util.Collection;

import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameterization.TrackedParameter;

/**
 * Result that keeps track of settings that were used in generating this
 * particular result.
 * 
 * @author Erich Schubert
 * @since 0.3
 */
public class SettingsResult extends BasicResult {
  /**
   * Settings storage.
   */
  Collection<TrackedParameter> settings;

  /**
   * Constructor.
   * 
   * @param settings Settings to store
   */
  public SettingsResult(Collection<TrackedParameter> settings) {
    super("Settings", "settings");
    this.settings = settings;
  }

  /**
   * Get the settings
   * 
   * @return the settings
   */
  public Collection<TrackedParameter> getSettings() {
    return settings;
  }
}
