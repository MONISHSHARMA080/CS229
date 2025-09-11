
import sys
import inspect
def printType(*args):
    print('====== variable info =====')
    
    # Get the calling frame
    frame = inspect.currentframe().f_back
    
    # Get the source code of the calling line
    try:
        # Get the line that called this function
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        
        # Read the source line
        with open(filename, 'r') as f:
            lines = f.readlines()
            call_line = lines[lineno - 1].strip()
        
        # Extract variable names from the function call
        # This is a simple approach - might need refinement for complex cases
        start = call_line.find('printType(') + len('printType(')
        end = call_line.rfind(')')
        args_str = call_line[start:end]
        var_names = [name.strip() for name in args_str.split(',')]
        
        # Print variable info
        for i, (name, value) in enumerate(zip(var_names, args)):
            if i < len(var_names):
                print(f'{name}:{type(value).__name__}', end=" ")
            else:
                print(f'arg{i}:{type(value).__name__}', end=" ")
        print('\n===== variable info =====')
        print('')
        
    except Exception as e:
        # Fallback: just print types with generic names
        for i, value in enumerate(args):
            print(f'arg{i}:{type(value).__name__}', end=" ")
        print('')
        print('==== variable info =====')
