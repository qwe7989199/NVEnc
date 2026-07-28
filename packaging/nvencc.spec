Name:       $PACKAGE_NAME$
Version:    $PACKAGE_VERSION$
Release:    1
Group:      Applications
Summary:    $PACKAGE_NAME$
Packager:   $PACKAGE_MAINTAINER$
License:    $PACKAGE_LICENSE$
Source:     tmp.tar.gz
BuildRoot:  %{_tmppath}/%{name}-%{version}-buildroot
BuildArch:  $PACKAGE_ARCH$
Requires:   $PACKAGE_DEPENDS$

%description
$PACKAGE_DESCRIPTION$

%global debug_package %{nil}

%prep
rm -rf $RPM_BUILD_ROOT

%setup -n %{name}

%build

%install
mkdir -p $RPM_BUILD_ROOT/usr/bin/
install -m 755 usr/bin/$PACKAGE_BIN$ $RPM_BUILD_ROOT/usr/bin/
mkdir -p $RPM_BUILD_ROOT%{_libdir}/
cp -a usr/lib64/libnvjpeg2k.so* $RPM_BUILD_ROOT%{_libdir}/
mkdir -p $RPM_BUILD_ROOT/usr/share/licenses/%{name}/
install -m 644 usr/share/licenses/%{name}/nvjpeg2000-LICENSE $RPM_BUILD_ROOT/usr/share/licenses/%{name}/

%clean
rm -rf $RPM_BUILD_ROOT

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%defattr(-, root, root)
/usr/bin/$PACKAGE_BIN$
%{_libdir}/libnvjpeg2k.so*
%license /usr/share/licenses/%{name}/nvjpeg2000-LICENSE

%changelog
