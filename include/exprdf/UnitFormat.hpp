#ifndef EXPRDF_UNIT_FORMAT_HPP_
#define EXPRDF_UNIT_FORMAT_HPP_

#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <complex>
#include <sstream>
#include <iomanip>

class UnitFormatter {
 public:
  // Quantity name constants
  static constexpr const char* kFrequency   = "frequency";
  static constexpr const char* kTime        = "time";
  static constexpr const char* kVoltage     = "voltage";
  static constexpr const char* kCurrent     = "current";
  static constexpr const char* kResistance  = "resistance";
  static constexpr const char* kCapacitance = "capacitance";
  static constexpr const char* kInductance  = "inductance";
  static constexpr const char* kPower       = "power";
  static constexpr const char* kLength      = "length";
  static constexpr const char* kBitrate     = "bitrate";

  struct Unit {
    std::string symbol;
    double factor;

    Unit(std::string sym = "", double f = 1.0)
        : symbol(std::move(sym)), factor(f) {}
  };

  struct FormatScale {
    double scale_factor;
    std::string unit_symbol;

    FormatScale(double scale = 1.0, std::string unit = "")
        : scale_factor(scale), unit_symbol(std::move(unit)) {}

    double Apply(double value) const { return value / scale_factor; }

    std::pair<double, std::string> GetFormatted(double value) const {
      return {Apply(value), unit_symbol};
    }
  };

  UnitFormatter(const UnitFormatter&) = delete;
  UnitFormatter& operator=(const UnitFormatter&) = delete;

  static UnitFormatter& Instance() {
    static UnitFormatter fmt;
    return fmt;
  }

  std::string FormatValue(const std::string& quantity, double value,
                          int precision = -1) const {
    FormatScale scale = FindScale(quantity, value);
    double scaled_value = scale.Apply(value);
    std::string number_str = FormatNumber(scaled_value, precision);
    return number_str + " " + scale.unit_symbol;
  }

  std::string FormatValue(const std::string& quantity, int value,
                          int precision = -1) const {
    return FormatValue(quantity, static_cast<double>(value), precision);
  }

  std::string FormatValue(const std::string& quantity,
                          std::complex<double> value,
                          int precision = -1) const {
    double mag = std::abs(value);
    FormatScale scale = FindScale(quantity, mag);
    double re = scale.Apply(value.real());
    double im = scale.Apply(value.imag());
    std::string re_str = FormatNumber(re, precision);
    std::string im_str = FormatNumber(std::abs(im), precision);

    std::string result;
    if (re != 0.0 && im != 0.0) {
      result = re_str + (im >= 0 ? " + " : " - ") + im_str + " j";
    } else if (im != 0.0) {
      result = (im >= 0 ? "" : "-") + im_str + " j";
    } else {
      result = re_str;
    }
    return result + " " + scale.unit_symbol;
  }

  FormatScale FindScale(const std::string& quantity, double value) const {
    auto it = unit_systems_.find(quantity);
    if (it == unit_systems_.end()) {
      return FormatScale(1.0, "?");
    }

    const auto& units = it->second;
    if (units.empty()) {
      return FormatScale(1.0, "?");
    }

    double abs_value = std::abs(value);
    if (abs_value == 0.0) {
      const Unit* smallest = &units[0];
      for (const auto& unit : units) {
        if (unit.factor < smallest->factor) smallest = &unit;
      }
      return FormatScale(smallest->factor, smallest->symbol);
    }

    const Unit* best_unit = &units[0];
    double best_score = std::abs(std::log10(abs_value / units[0].factor));

    for (const auto& unit : units) {
      double normalized = abs_value / unit.factor;

      if (normalized >= 1.0 && normalized < 1000.0) {
        double n_best = abs_value / best_unit->factor;
        if (n_best < 1.0 || n_best >= 1000.0 ||
            unit.factor > best_unit->factor) {
          best_unit = &unit;
        }
        continue;
      }

      double n_best = abs_value / best_unit->factor;
      if (n_best >= 1.0 && n_best < 1000.0) continue;

      double score = std::abs(std::log10(normalized));
      if (score < best_score) {
        best_score = score;
        best_unit = &unit;
      }
    }

    return FormatScale(best_unit->factor, best_unit->symbol);
  }

  void RegisterUnit(const std::string& quantity,
                    const std::vector<Unit>& units) {
    unit_systems_[quantity] = units;
  }

 private:
  UnitFormatter() { InitDefaultUnits(); }

  std::unordered_map<std::string, std::vector<Unit>> unit_systems_;

  void InitDefaultUnits() {
    unit_systems_[kFrequency] = {
        {"Hz", 1.0}, {"kHz", 1e3}, {"MHz", 1e6}, {"GHz", 1e9}, {"THz", 1e12}};

    unit_systems_[kTime] = {{"s", 1.0},    {"ms", 1e-3},  {"us", 1e-6},
                            {"ns", 1e-9},  {"ps", 1e-12}, {"fs", 1e-15}};

    unit_systems_[kVoltage] = {
        {"V", 1.0}, {"kV", 1e3}, {"mV", 1e-3}, {"uV", 1e-6}, {"nV", 1e-9}};

    unit_systems_[kCurrent] = {{"A", 1.0},   {"kA", 1e3},   {"mA", 1e-3},
                               {"uA", 1e-6}, {"nA", 1e-9},  {"pA", 1e-12}};

    unit_systems_[kResistance] = {
        {"ohm", 1.0}, {"kohm", 1e3}, {"Mohm", 1e6}};

    unit_systems_[kCapacitance] = {{"F", 1.0},   {"mF", 1e-3},  {"uF", 1e-6},
                                   {"nF", 1e-9}, {"pF", 1e-12}, {"fF", 1e-15}};

    unit_systems_[kInductance] = {{"H", 1.0},   {"mH", 1e-3},  {"uH", 1e-6},
                                  {"nH", 1e-9}, {"pH", 1e-12}};

    unit_systems_[kPower] = {
        {"W", 1.0}, {"kW", 1e3}, {"mW", 1e-3}, {"uW", 1e-6}, {"nW", 1e-9}};

    unit_systems_[kLength] = {
        {"m", 1.0}, {"mm", 1e-3}, {"um", 1e-6}, {"nm", 1e-9}, {"pm", 1e-12}};

    unit_systems_[kBitrate] = {{"bps", 1.0},  {"kbps", 1e3},  {"Mbps", 1e6},
                               {"Gbps", 1e9}, {"Tbps", 1e12}};
  }

  std::string FormatNumber(double value, int precision = -1) const {
    double abs_value = std::abs(value);

    if (abs_value == 0.0) {
      return "0";
    }

    if (precision >= 0) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(precision) << value;
      return oss.str();
    }

    if (abs_value >= 1000 || abs_value < 0.001) {
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6) << value;
      return oss.str();
    } else if (abs_value >= 100) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(1) << value;
      return oss.str();
    } else if (abs_value >= 10) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << value;
      return oss.str();
    } else {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3) << value;
      return oss.str();
    }
  }
};

// ============================================================
// unit_format — public API
// ============================================================
namespace unit_format {

using Unit        = UnitFormatter::Unit;
using FormatScale = UnitFormatter::FormatScale;

constexpr const char* kFrequency   = UnitFormatter::kFrequency;
constexpr const char* kTime        = UnitFormatter::kTime;
constexpr const char* kVoltage     = UnitFormatter::kVoltage;
constexpr const char* kCurrent     = UnitFormatter::kCurrent;
constexpr const char* kResistance  = UnitFormatter::kResistance;
constexpr const char* kCapacitance = UnitFormatter::kCapacitance;
constexpr const char* kInductance  = UnitFormatter::kInductance;
constexpr const char* kPower       = UnitFormatter::kPower;
constexpr const char* kLength      = UnitFormatter::kLength;
constexpr const char* kBitrate     = UnitFormatter::kBitrate;

inline std::string Format(const std::string& quantity, double value,
                          int precision = -1) {
  return UnitFormatter::Instance().FormatValue(quantity, value, precision);
}

inline std::string Format(const std::string& quantity, int value,
                          int precision = -1) {
  return UnitFormatter::Instance().FormatValue(quantity, value, precision);
}

inline std::string Format(const std::string& quantity,
                          std::complex<double> value, int precision = -1) {
  return UnitFormatter::Instance().FormatValue(quantity, value, precision);
}

inline FormatScale GetFormatScale(const std::string& quantity, double value) {
  return UnitFormatter::Instance().FindScale(quantity, value);
}

inline FormatScale GetRangeFormatScale(const std::string& quantity,
                                       double start, double stop) {
  double max_abs = std::max(std::abs(start), std::abs(stop));
  return UnitFormatter::Instance().FindScale(quantity, max_abs);
}

inline void RegisterUnit(const std::string& quantity,
                         const std::vector<Unit>& units) {
  UnitFormatter::Instance().RegisterUnit(quantity, units);
}

inline std::string Freq(double value, int precision = -1) {
  return Format(kFrequency, value, precision);
}
inline std::string Duration(double value, int precision = -1) {
  return Format(kTime, value, precision);
}
inline std::string Volt(double value, int precision = -1) {
  return Format(kVoltage, value, precision);
}
inline std::string Amp(double value, int precision = -1) {
  return Format(kCurrent, value, precision);
}
inline std::string Ohm(double value, int precision = -1) {
  return Format(kResistance, value, precision);
}
inline std::string Farad(double value, int precision = -1) {
  return Format(kCapacitance, value, precision);
}
inline std::string Henry(double value, int precision = -1) {
  return Format(kInductance, value, precision);
}

}  // namespace unit_format

#endif  // EXPRDF_UNIT_FORMAT_HPP_
