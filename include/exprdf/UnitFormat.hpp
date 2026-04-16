#ifndef EXPRDF_UNIT_FORMAT_HPP_
#define EXPRDF_UNIT_FORMAT_HPP_

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace unit_format {

namespace quantity {

constexpr const char unitless[] = "";
constexpr const char frequency[] = "frequency";
constexpr const char time[] = "time";
constexpr const char voltage[] = "voltage";
constexpr const char current[] = "current";
constexpr const char resistance[] = "resistance";
constexpr const char capacitance[] = "capacitance";
constexpr const char inductance[] = "inductance";
constexpr const char power[] = "power";
constexpr const char length[] = "length";
constexpr const char bitrate[] = "bitrate";

}  // namespace quantity

struct UnitDefinition {
  std::string symbol;
  double factor;

  UnitDefinition(std::string unit_symbol = "", double unit_factor = 1.0)
      : symbol(std::move(unit_symbol)), factor(unit_factor) {}
};

struct Scale {
  double scale_factor;
  std::string unit_symbol;

  Scale(double factor = 1.0, std::string symbol = "")
      : scale_factor(factor), unit_symbol(std::move(symbol)) {}

  double apply(double value) const { return value / scale_factor; }

  std::pair<double, std::string> format(double value) const {
    return std::make_pair(apply(value), unit_symbol);
  }

  double Apply(double value) const { return apply(value); }

  std::pair<double, std::string> GetFormatted(double value) const {
    return format(value);
  }
};

typedef UnitDefinition Unit;
typedef Scale FormatScale;

namespace detail {

class Registry {
 public:
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;

  static Registry& instance() {
    static Registry registry;
    return registry;
  }

  std::string format_value(const std::string& quantity_name, double value,
                           int precision = -1) const {
    Scale scale = select_scale(quantity_name, value);
    const double scaled_value = scale.apply(value);
    const std::string number = format_number(scaled_value, precision);
    if (scale.unit_symbol.empty()) {
      return number;
    }
    return number + " " + scale.unit_symbol;
  }

  std::string format_value(const std::string& quantity_name, int value,
                           int precision = -1) const {
    return format_value(quantity_name, static_cast<double>(value), precision);
  }

  std::string format_value(const std::string& quantity_name,
                           std::complex<double> value,
                           int precision = -1) const {
    const Scale scale = select_scale(quantity_name, std::abs(value));
    const double real = scale.apply(value.real());
    const double imag = scale.apply(value.imag());
    const std::string real_str = format_number(real, precision);
    const std::string imag_str = format_number(std::abs(imag), precision);

    std::string result;
    if (real != 0.0 && imag != 0.0) {
      result = real_str + (imag >= 0 ? " + " : " - ") + imag_str + " j";
    } else if (imag != 0.0) {
      result = (imag >= 0 ? "" : "-") + imag_str + " j";
    } else {
      result = real_str;
    }
    if (scale.unit_symbol.empty()) {
      return result;
    }
    return result + " " + scale.unit_symbol;
  }

  Scale base_scale(const std::string& quantity_name) const {
    const auto it = unit_systems_.find(quantity_name);
    if (it == unit_systems_.end() || it->second.empty()) {
      return Scale(1.0, "?");
    }
    const std::vector<UnitDefinition>& units = it->second;
    const UnitDefinition* base_unit = &units[0];
    for (const auto& unit : units) {
      if (unit.factor == 1.0) {
        base_unit = &unit;
        break;
      }
    }
    return Scale(base_unit->factor, base_unit->symbol);
  }

  Scale select_scale(const std::string& quantity_name, double value) const {
    const auto it = unit_systems_.find(quantity_name);
    if (it == unit_systems_.end() || it->second.empty()) {
      return Scale(1.0, "?");
    }

    const std::vector<UnitDefinition>& units = it->second;
    const double abs_value = std::abs(value);
    if (abs_value == 0.0) {
      return base_scale(quantity_name);
    }

    const UnitDefinition* best_unit = &units[0];
    double best_score = std::abs(std::log10(abs_value / units[0].factor));

    for (const auto& unit : units) {
      const double normalized = abs_value / unit.factor;
      const double current_normalized = abs_value / best_unit->factor;

      if (normalized >= 1.0 && normalized < 1000.0) {
        if (current_normalized < 1.0 || current_normalized >= 1000.0 ||
            unit.factor > best_unit->factor) {
          best_unit = &unit;
        }
        continue;
      }

      if (current_normalized >= 1.0 && current_normalized < 1000.0) {
        continue;
      }

      const double score = std::abs(std::log10(normalized));
      if (score < best_score) {
        best_score = score;
        best_unit = &unit;
      }
    }

    return Scale(best_unit->factor, best_unit->symbol);
  }

  void register_units(const std::string& quantity_name,
                      const std::vector<UnitDefinition>& units) {
    unit_systems_[quantity_name] = units;
  }

 private:
  Registry() { init_default_units(); }

  std::unordered_map<std::string, std::vector<UnitDefinition>> unit_systems_;

  void init_default_units() {
    register_units(quantity::unitless, {{"", 1.0}});
    register_units(quantity::frequency,
                   {{"Hz", 1.0}, {"kHz", 1e3}, {"MHz", 1e6},
                    {"GHz", 1e9}, {"THz", 1e12}});
    register_units(quantity::time,
                   {{"s", 1.0}, {"ms", 1e-3}, {"us", 1e-6},
                    {"ns", 1e-9}, {"ps", 1e-12}, {"fs", 1e-15}});
    register_units(quantity::voltage,
                   {{"V", 1.0}, {"kV", 1e3}, {"mV", 1e-3},
                    {"uV", 1e-6}, {"nV", 1e-9}});
    register_units(quantity::current,
                   {{"A", 1.0}, {"kA", 1e3}, {"mA", 1e-3},
                    {"uA", 1e-6}, {"nA", 1e-9}, {"pA", 1e-12}});
    register_units(quantity::resistance,
                   {{"ohm", 1.0}, {"kohm", 1e3}, {"Mohm", 1e6}});
    register_units(quantity::capacitance,
                   {{"F", 1.0}, {"mF", 1e-3}, {"uF", 1e-6},
                    {"nF", 1e-9}, {"pF", 1e-12}, {"fF", 1e-15}});
    register_units(quantity::inductance,
                   {{"H", 1.0}, {"mH", 1e-3}, {"uH", 1e-6},
                    {"nH", 1e-9}, {"pH", 1e-12}});
    register_units(quantity::power,
                   {{"W", 1.0}, {"kW", 1e3}, {"mW", 1e-3},
                    {"uW", 1e-6}, {"nW", 1e-9}});
    register_units(quantity::length,
                   {{"m", 1.0}, {"mm", 1e-3}, {"um", 1e-6},
                    {"nm", 1e-9}, {"pm", 1e-12}});
    register_units(quantity::bitrate,
                   {{"bps", 1.0}, {"kbps", 1e3}, {"Mbps", 1e6},
                    {"Gbps", 1e9}, {"Tbps", 1e12}});
  }

  std::string format_number(double value, int precision = -1) const {
    const double abs_value = std::abs(value);
    if (abs_value == 0.0) {
      return "0";
    }

    std::ostringstream output;
    if (precision >= 0) {
      output << std::fixed << std::setprecision(precision) << value;
    } else if (abs_value >= 1000 || abs_value < 0.001) {
      output << std::scientific << std::setprecision(6) << value;
    } else if (abs_value >= 100) {
      output << std::fixed << std::setprecision(1) << value;
    } else if (abs_value >= 10) {
      output << std::fixed << std::setprecision(2) << value;
    } else {
      output << std::fixed << std::setprecision(3) << value;
    }
    return output.str();
  }
};

}  // namespace detail

inline std::string format(const std::string& quantity_name, double value,
                          int precision = -1) {
  return detail::Registry::instance().format_value(quantity_name, value,
                                                   precision);
}

inline std::string format(const std::string& quantity_name, int value,
                          int precision = -1) {
  return detail::Registry::instance().format_value(quantity_name, value,
                                                   precision);
}

inline std::string format(const std::string& quantity_name,
                          std::complex<double> value,
                          int precision = -1) {
  return detail::Registry::instance().format_value(quantity_name, value,
                                                   precision);
}

inline Scale scale_for(const std::string& quantity_name, double value) {
  return detail::Registry::instance().select_scale(quantity_name, value);
}

inline Scale base_scale(const std::string& quantity_name) {
  return detail::Registry::instance().base_scale(quantity_name);
}

inline Scale scale_for_range(const std::string& quantity_name, double start,
                             double stop) {
  return scale_for(quantity_name, std::max(std::abs(start), std::abs(stop)));
}

inline void register_units(const std::string& quantity_name,
                           const std::vector<UnitDefinition>& units) {
  detail::Registry::instance().register_units(quantity_name, units);
}

inline std::string format_frequency(double value, int precision = -1) {
  return format(quantity::frequency, value, precision);
}

inline std::string format_duration(double value, int precision = -1) {
  return format(quantity::time, value, precision);
}

inline std::string format_voltage(double value, int precision = -1) {
  return format(quantity::voltage, value, precision);
}

inline std::string format_current(double value, int precision = -1) {
  return format(quantity::current, value, precision);
}

inline std::string format_resistance(double value, int precision = -1) {
  return format(quantity::resistance, value, precision);
}

inline std::string format_capacitance(double value, int precision = -1) {
  return format(quantity::capacitance, value, precision);
}

inline std::string format_inductance(double value, int precision = -1) {
  return format(quantity::inductance, value, precision);
}

inline std::string Format(const std::string& quantity_name, double value,
                          int precision = -1) {
  return format(quantity_name, value, precision);
}

inline std::string Format(const std::string& quantity_name, int value,
                          int precision = -1) {
  return format(quantity_name, value, precision);
}

inline std::string Format(const std::string& quantity_name,
                          std::complex<double> value,
                          int precision = -1) {
  return format(quantity_name, value, precision);
}

inline FormatScale GetFormatScale(const std::string& quantity_name,
                                  double value) {
  return scale_for(quantity_name, value);
}

inline FormatScale GetRangeFormatScale(const std::string& quantity_name,
                                       double start, double stop) {
  return scale_for_range(quantity_name, start, stop);
}

inline void RegisterUnit(const std::string& quantity_name,
                         const std::vector<Unit>& units) {
  register_units(quantity_name, units);
}

inline std::string Freq(double value, int precision = -1) {
  return format_frequency(value, precision);
}

inline std::string Duration(double value, int precision = -1) {
  return format_duration(value, precision);
}

inline std::string Volt(double value, int precision = -1) {
  return format_voltage(value, precision);
}

inline std::string Amp(double value, int precision = -1) {
  return format_current(value, precision);
}

inline std::string Ohm(double value, int precision = -1) {
  return format_resistance(value, precision);
}

inline std::string Farad(double value, int precision = -1) {
  return format_capacitance(value, precision);
}

inline std::string Henry(double value, int precision = -1) {
  return format_inductance(value, precision);
}

}  // namespace unit_format

#endif  // EXPRDF_UNIT_FORMAT_HPP_
