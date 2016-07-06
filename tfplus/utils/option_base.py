import cmd_args
from option_saver import OptionSaver


class OptionBase(object):

    def __init__(self):
        self._opt = {}
        self._default_opt = {}
        self._opt_names = set()
        self.init_default_options()
        pass

    def init_default_options(self):
        pass

    def set_option(self, name, value):
        """Set an option.

        Args:
            name: option key
            value: option value

        Returns:
            self
        """
        self._opt[name] = value
        return self

    def set_all_options(self, opt):
        """Set all options.

        Args:
            opt: dictionary
        """
        self._opt = opt
        return self

    def register_option(self, name):
        """Register an option."""
        self._opt_names.add(name)
        pass

    def get_option(self, name):
        """Get an option.

        Args:
            name: option key

        Returns:
            value: option value
        """
        if name in self._opt:
            return self._opt[name]
        else:
            return self.get_default_option(name)

    def get_all_options(self):
        return self._opt

    def read_options(self, folder, name):
        return OptionSaver(folder, name).read()

    def get_options_for_save(self):
        return self._opt

    def save_options(self, folder, name):
        OptionSaver(folder, name).save(self.get_options_for_save())
        pass

    def get_default_option(self, name):
        """Get default option.

        Args:
            name: option key

        Returns:
            value: option value
        """
        return self._default_opt[name]

    def set_default_option(self, name, value):
        """Add default option.

        Args:
            name: option key

        Returns:
            value: option value
        """
        self._default_opt[name] = value
        pass

    def get_description(self):
        return 'None'

    def parse_opt(self):
        opt = cmd_args.make()
        for key in opt.keys():
            if key not in self._opt_names:
                del opt[key]
                pass
            pass
        return opt

    def init_from_main(self):
        opt = self.parse_opt()
        return self.set_all_options(opt)
    pass
