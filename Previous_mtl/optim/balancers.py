class BalancerRegistry:
    registry = {}


def get_method(method, *args, **kwargs):
    
    print(f'view regisrty {BalancerRegistry.registry}')
    if method not in BalancerRegistry.registry:
        raise ValueError("Balancer named '{}' is not defined, valid methods are: {}".format(
            method, ', '.join(BalancerRegistry.registry.keys())))
    method_cls, method_args, method_kwargs = BalancerRegistry.registry[method]
    # print method_cls, method_args, method_kwargs
    print(f'view method_cls {method_cls}')
    print(f'view method_args {method_args}')
    print(f'view method_kwargs {method_kwargs}')
    return method_cls(*method_args, *args, **method_kwargs, **kwargs)


def register(name, *args, **kwargs):
    def _register(cls):
        print(f'    view name {name}')
        print(f'    view cls {cls}')
        BalancerRegistry.registry[name] = (cls, args, kwargs)
        return cls
    return _register
