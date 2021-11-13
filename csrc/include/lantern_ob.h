template <class T>
class LanternObject
{
private:
  T _object;
  
public:
  LanternObject(T object) : _object(std::forward<T>(object))
  {
  }
  
  LanternObject()
  {
  }
  
  T &get()
  {
    return _object;
  }
};
