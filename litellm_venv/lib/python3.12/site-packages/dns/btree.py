# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

"""
A BTree in the style of Cormen, Leiserson, and Rivest's "Algorithms" book, with
copy-on-write node updates, cursors, and optional space optimization for mostly-in-order
insertion.
"""

from collections.abc import MutableMapping, MutableSet
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar, cast

DEFAULT_T = 127

KT = TypeVar("KT")  # the type of a key in Element


class Element(Generic[KT]):
    """All items stored in the BTree are Elements."""

    def key(self) -> KT:
        """The key for this element; the returned type must implement comparison."""
        raise NotImplementedError  # pragma: no cover


ET = TypeVar("ET", bound=Element)  # the type of a value in a _KV


def _MIN(t: int) -> int:
    """The minimum number of keys in a non-root node for a BTree with the specified
    ``t``
    """
    return t - 1


def _MAX(t: int) -> int:
    """The maximum number of keys in node for a BTree with the specified ``t``"""
    return 2 * t - 1


class _Creator:
    """A _Creator class instance is used as a unique id for the BTree which created
    a node.

    We use a dedicated creator rather than just a BTree reference to avoid circularity
    that would complicate GC.
    """

    def __str__(self):  # pragma: no cover
        return f"{id(self):x}"


class _Node(Generic[KT, ET]):
    """A Node in the BTree.

    A Node (leaf or internal) of the BTree.
    """

    __slots__ = ["t", "creator", "is_leaf", "elts", "children"]

    def __init__(self, t: int, creator: _Creator, is_leaf: bool):
        assert t >= 3
        self.t = t
        self.creator = creator
        self.is_leaf = is_leaf
        self.elts: list[ET] = []
        self.children: list[_Node[KT, ET]] = []

    def is_maximal(self) -> bool:
        """Does this node have the maximal number of keys?"""
        assert len(self.elts) <= _MAX(self.t)
        return len(self.elts) == _MAX(self.t)

    def is_minimal(self) -> bool:
        """Does this node have the minimal number of keys?"""
        assert len(self.elts) >= _MIN(self.t)
        return len(self.elts) == _MIN(self.t)

    def search_in_node(self, key: KT) -> tuple[int, bool]:
        """Get the index of the ``Element`` matching ``key`` or the index of its
        least successor.

        Returns a tuple of the index and an ``equal`` boolean that is ``True`` iff.
        the key was found.
        """
        l = len(self.elts)
        if l > 0 and key > self.elts[l - 1].key():
            # This is optimizing near in-order insertion.
            return l, False
        l = 0
        i = len(self.elts)
        r = i - 1
        equal = False
        while l <= r:
            m = (l + r) // 2
            k = self.elts[m].key()
            if key == k:
                i = m
                equal = True
                break
            elif key < k:
                i = m
                r = m - 1
            else:
                l = m + 1
        return i, equal

    def maybe_cow_child(self, index: int) -> "_Node[KT, ET]":
        assert not self.is_leaf
        child = self.children[index]
        cloned = child.maybe_cow(self.creator)
        if cloned:
            self.children[index] = cloned
            return cloned
        else:
            return child

    def _get_node(self, key: KT) -> Tuple[Optional["_Node[KT, ET]"], int]:
        """Get the node associated with key and its index, doing
        copy-on-write if we have to descend.

        Returns a tuple of the node and the index, or the tuple ``(None, 0)``
        if the key was not found.
        """
        i, equal = self.search_in_node(key)
        if equal:
            return (self, i)
        elif self.is_leaf:
            return (None, 0)
        else:
            child = self.maybe_cow_child(i)
            return child._get_node(key)

    def get(self, key: KT) -> ET | None:
        """Get the element associated with *key* or return ``None``"""
        i, equal = self.search_in_node(key)
        if equal:
            return self.elts[i]
        elif self.is_leaf:
            return None
        else:
            return self.children[i].get(key)

    def optimize_in_order_insertion(self, index: int) -> None:
        """Try to minimize the number of Nodes in a BTree where the insertion
        is done in-order or close to it, by stealing as much as we can from our
        right sibling.

        If we don't do this, then an in-order insertion will produce a BTree
        where most of the nodes are minimal.
        """
        if index == 0:
            return
        left = self.children[index - 1]
        if len(left.elts) == _MAX(self.t):
            return
        left = self.maybe_cow_child(index - 1)
        while len(left.elts) < _MAX(self.t):
            if not left.try_right_steal(self, index - 1):
                break

    def insert_nonfull(self, element: ET, in_order: bool) -> ET | None:
        assert not self.is_maximal()
        while True:
            key = element.key()
            i, equal = self.search_in_node(key)
            if equal:
                # replace
                old = self.elts[i]
                self.elts[i] = element
                return old
            elif self.is_leaf:
                self.elts.insert(i, element)
                return None
            else:
                child = self.maybe_cow_child(i)
                if child.is_maximal():
                    self.adopt(*child.split())
                    # Splitting might result in our target moving to us, so
                    # search again.
                    continue
                oelt = child.insert_nonfull(element, in_order)
                if in_order:
                    self.optimize_in_order_insertion(i)
                return oelt

    def split(self) -> tuple["_Node[KT, ET]", ET, "_Node[KT, ET]"]:
        """Split a maximal node into two minimal ones and a central element."""
        assert self.is_maximal()
        right = self.__class__(self.t, self.creator, self.is_leaf)
        right.elts = list(self.elts[_MIN(self.t) + 1 :])
        middle = self.elts[_MIN(self.t)]
        self.elts = list(self.elts[: _MIN(self.t)])
        if not self.is_leaf:
            right.children = list(self.children[_MIN(self.t) + 1 :])
            self.children = list(self.children[: _MIN(self.t) + 1])
        return self, middle, right

    def try_left_steal(self, parent: "_Node[KT, ET]", index: int) -> bool:
        """Try to steal from this Node's left sibling for balancing purposes.

        Returns ``True`` if the theft was successful, or ``False`` if not.
        """
        if index != 0:
            left = parent.children[index - 1]
            if not left.is_minimal():
                left = parent.maybe_cow_child(index - 1)
                elt = parent.elts[index - 1]
                parent.elts[index - 1] = left.elts.pop()
                self.elts.insert(0, elt)
                if not left.is_leaf:
                    assert not self.is_leaf
                    child = left.children.pop()
                    self.children.insert(0, child)
                return True
        return False

    def try_right_steal(self, parent: "_Node[KT, ET]", index: int) -> bool:
        """Try to steal from this Node's right sibling for balancing purposes.

        Returns ``True`` if the theft was successful, or ``False`` if not.
        """
        if index + 1 < len(parent.children):
            right = parent.children[index + 1]
            if not right.is_minimal():
                right = parent.maybe_cow_child(index + 1)
                elt = parent.elts[index]
                parent.elts[index] = right.elts.pop(0)
                self.elts.append(elt)
                if not right.is_leaf:
                    assert not self.is_leaf
                    child = right.children.pop(0)
                    self.children.append(child)
                return True
        return False

    def adopt(self, left: "_Node[KT, ET]", middle: ET, right: "_Node[KT, ET]") -> None:
        """Adopt left, middle, and right into our Node (which must not be maximal,
        and which must not be a leaf).  In the case were we are not the new root,
        then the left child must already be in the Node."""
        assert not self.is_maximal()
        assert not self.is_leaf
        key = middle.key()
        i, equal = self.search_in_node(key)
        assert not equal
        self.elts.insert(i, middle)
        if len(self.children) == 0:
            # We are the new root
            self.children = [left, right]
        else:
            assert self.children[i] == left
            self.children.insert(i + 1, right)

    def merge(self, parent: "_Node[KT, ET]", index: int) -> None:
        """Merge this node's parent and its right sibling into this node."""
        right = parent.children.pop(index + 1)
        self.elts.append(parent.elts.pop(index))
        self.elts.extend(right.elts)
        if not self.is_leaf:
            self.children.extend(right.children)

    def minimum(self) -> ET:
        """The least element in this subtree."""
        if self.is_leaf:
            return self.elts[0]
        else:
            return self.children[0].minimum()

    def maximum(self) -> ET:
        """The greatest element in this subtree."""
        if self.is_leaf:
            return self.elts[-1]
        else:
            return self.children[-1].maximum()

    def balance(self, parent: "_Node[KT, ET]", index: int) -> None:
        """This Node is minimal, and we want to make it non-minimal so we can delete.
        We try to steal from our siblings, and if that doesn't work we will merge
        with one of them."""
        assert not parent.is_leaf
        if self.try_left_steal(parent, index):
            return
        if self.try_right_steal(parent, index):
            return
        # Stealing didn't work, so both siblings must be minimal.
        if index == 0:
            # We are the left-most node so merge with our right sibling.
            self.merge(parent, index)
        else:
            # Have our left sibling merge with us.  This lets us only have "merge right"
            # code.
            left = parent.maybe_cow_child(index - 1)
            left.merge(parent, index - 1)

    def delete(
        self, key: KT, parent: Optional["_Node[KT, ET]"], exact: ET | None
    ) -> ET | None:
        """Delete an element matching *key* if it exists.  If *exact* is not ``None``
        then it must be an exact match with that element.  The Node must not be
        minimal unless it is the root."""
        assert parent is None or not self.is_minimal()
        i, equal = self.search_in_node(key)
        original_key = None
        if equal:
            # Note we use "is" here as we meant "exactly this object".
            if exact is not None and self.elts[i] is not exact:
                raise ValueError("exact delete did not match existing elt")
            if self.is_leaf:
                return self.elts.pop(i)
            # Note we need to ensure exact is None going forward as we've
            # already checked exactness and are about to change our target key
            # to the least successor.
            exact = None
            original_key = key
            least_successor = self.children[i + 1].minimum()
            key = least_successor.key()
            i = i + 1
        if self.is_leaf:
            # No match
            if exact is not None:
                raise ValueError("exact delete had no match")
            return None
        # recursively delete in the appropriate child
        child = self.maybe_cow_child(i)
        if child.is_minimal():
            child.balance(self, i)
            # Things may have moved.
            i, equal = self.search_in_node(key)
            assert not equal
            child = self.children[i]
            assert not child.is_minimal()
        elt = child.delete(key, self, exact)
        if original_key is not None:
            node, i = self._get_node(original_key)
            assert node is not None
            assert elt is not None
            oelt = node.elts[i]
            node.elts[i] = elt
            elt = oelt
        return elt

    def visit_in_order(self, visit: Callable[[ET], None]) -> None:
        """Call *visit* on all of the elements in order."""
        for i, elt in enumerate(self.elts):
            if not self.is_leaf:
                self.children[i].visit_in_order(visit)
            visit(elt)
        if not self.is_leaf:
            self.children[-1].visit_in_order(visit)

    def _visit_preorder_by_node(self, visit: Callable[["_Node[KT, ET]"], None]) -> None:
        """Visit nodes in preorder.  This method is only used for testing."""
        visit(self)
        if not self.is_leaf:
            for child in self.children:
                child._visit_preorder_by_node(visit)

    def maybe_cow(self, creator: _Creator) -> Optional["_Node[KT, ET]"]:
        """Return a clone of this Node if it was not created by *creator*, or ``None``
        otherwise (i.e. copy for copy-on-write if we haven't already copied it)."""
        if self.creator is not creator:
            return self.clone(creator)
        else:
            return None

    def clone(self, creator: _Creator) -> "_Node[KT, ET]":
        """Make a shallow-copy duplicate of this node."""
        cloned = self.__class__(self.t, creator, self.is_leaf)
        cloned.elts.extend(self.elts)
        if not self.is_leaf:
            cloned.children.extend(self.children)
        return cloned

    def __str__(self):  # pragma: no cover
        if not self.is_leaf:
            children = " " + " ".join([f"{id(c):x}" for c in self.children])
        else:
            children = ""
        return f"{id(self):x} {self.creator} {self.elts}{children}"


class Cursor(Generic[KT, ET]):
    """A seekable cursor for a BTree.

    If you are going to use a cursor on a mutable BTree, you should use it
    in a ``with`` block so that any mutations of the BTree automatically park
    the cursor.
    """

    def __init__(self, btree: "BTree[KT, ET]"):
        self.btree = btree
        self.current_node: _Node | None = None
        # The current index is the element index within the current node, or
        # if there is no current node then it is 0 on the left boundary and 1
        # on the right boundary.
        self.current_index: int = 0
        self.recurse = False
        self.increasing = True
        self.parents: list[tuple[_Node, int]] = []
        self.parked = False
        self.parking_key: KT | None = None
        self.parking_key_read = False

    def _seek_least(self) -> None:
        # seek to the least value in the subtree beneath the current index of the
        # current node
        assert self.current_node is not None
        while not self.current_node.is_leaf:
            self.parents.append((self.current_node, self.current_index))
            self.current_node = self.current_node.children[self.current_index]
            assert self.current_node is not None
            self.current_index = 0

    def _seek_greatest(self) -> None:
        # seek to the greatest value in the subtree beneath the current index of the
        # current node
        assert self.current_node is not None
        while not self.current_node.is_leaf:
            self.parents.append((self.current_node, self.current_index))
            self.current_node = self.current_node.children[self.current_index]
            assert self.current_node is not None
            self.current_index = len(self.current_node.elts)

    def park(self):
        """Park the cursor.

        A cursor must be "parked" before mutating the BTree to avoid undefined behavior.
        Cursors created in a ``with`` block register with their BTree and will park
        automatically.  Note that a parked cursor may not observe some changes made when
        it is parked; for example a cursor being iterated with next() will not see items
        inserted before its current position.
        """
        if not self.parked:
            self.parked = True

    def _maybe_unpark(self):
        if self.parked:
            if self.parking_key is not None:
                # remember our increasing hint, as seeking might change it
                increasing = self.increasing
                if self.parking_key_read:
                    # We've already returned the parking key, so we want to be before it
                    # if decreasing and after it if increasing.
                    before = not self.increasing
                else:
                    # We haven't returned the parking key, so we've parked right
                    # after seeking or are on a boundary.  Either way, the before
                    # hint we want is the value of self.increasing.
                    before = self.increasing
                self.seek(self.parking_key, before)
                self.increasing = increasing  # might have been altered by seek()
            self.parked = False
            self.parking_key = None

    def prev(self) -> ET | None:
        """Get the previous element, or return None if on the left boundary."""
        self._maybe_unpark()
        self.parking_key = None
        if self.current_node is None:
            # on a boundary
            if self.current_index == 0:
                # left boundary, there is no prev
                return None
            else:
                assert self.current_index == 1
                # right boundary; seek to the actual boundary
                # so we can do a prev()
                self.current_node = self.btree.root
                self.current_index = len(self.btree.root.elts)
                self._seek_greatest()
        while True:
            if self.recurse:
                if not self.increasing:
                    # We only want to recurse if we are continuing in the decreasing
                    # direction.
                    self._seek_greatest()
                self.recurse = False
            self.increasing = False
            self.current_index -= 1
            if self.current_index >= 0:
                elt = self.current_node.elts[self.current_index]
                if not self.current_node.is_leaf:
                    self.recurse = True
                self.parking_key = elt.key()
                self.parking_key_read = True
                return elt
            else:
                if len(self.parents) > 0:
                    self.current_node, self.current_index = self.parents.pop()
                else:
                    self.current_node = None
                    self.current_index = 0
                    return None

    def next(self) -> ET | None:
        """Get the next element, or return None if on the right boundary."""
        self._maybe_unpark()
        self.parking_key = None
        if self.current_node is None:
            # on a boundary
            if self.current_index == 1:
                # right boundary, there is no next
                return None
            else:
                assert self.current_index == 0
                # left boundary; seek to the actual boundary
                # so we can do a next()
                self.current_node = self.btree.root
                self.current_index = 0
                self._seek_least()
        while True:
            if self.recurse:
                if self.increasing:
                    # We only want to recurse if we are continuing in the increasing
                    # direction.
                    self._seek_least()
                self.recurse = False
            self.increasing = True
            if self.current_index < len(self.current_node.elts):
                elt = self.current_node.elts[self.current_index]
                self.current_index += 1
                if not self.current_node.is_leaf:
                    self.recurse = True
                self.parking_key = elt.key()
                self.parking_key_read = True
                return elt
            else:
                if len(self.parents) > 0:
                    self.current_node, self.current_index = self.parents.pop()
                else:
                    self.current_node = None
                    self.current_index = 1
                    return None

    def _adjust_for_before(self, before: bool, i: int) -> None:
        if before:
            self.current_index = i
        else:
            self.current_index = i + 1

    def seek(self, key: KT, before: bool = True) -> None:
        """Seek to the specified key.

        If *before* is ``True`` (the default) then the cursor is positioned just
        before *key* if it exists, or before its least successor if it doesn't.  A
        subsequent next() will retrieve this value.  If *before* is ``False``, then
        the cursor is positioned just after *key* if it exists, or its greatest
        precessessor if it doesn't.  A subsequent prev() will return this value.
        """
        self.current_node = self.btree.root
        assert self.current_node is not None
        self.recurse = False
        self.parents = []
        self.increasing = before
        self.parked = False
        self.parking_key = key
        self.parking_key_read = False
        while not self.current_node.is_leaf:
            i, equal = self.current_node.search_in_node(key)
            if equal:
                self._adjust_for_before(before, i)
                if before:
                    self._seek_greatest()
                else:
                    self._seek_least()
                return
            self.parents.append((self.current_node, i))
            self.current_node = self.current_node.children[i]
            assert self.current_node is not None
        i, equal = self.current_node.search_in_node(key)
        if equal:
            self._adjust_for_before(before, i)
        else:
            self.current_index = i

    def seek_first(self) -> None:
        """Seek to the left boundary (i.e. just before the least element).

        A subsequent next() will return the least element if the BTree isn't empty."""
        self.current_node = None
        self.current_index = 0
        self.recurse = False
        self.increasing = True
        self.parents = []
        self.parked = False
        self.parking_key = None

    def seek_last(self) -> None:
        """Seek to the right boundary (i.e. just after the greatest element).

        A subsequent prev() will return the greatest element if the BTree isn't empty.
        """
        self.current_node = None
        self.current_index = 1
        self.recurse = False
        self.increasing = False
        self.parents = []
        self.parked = False
        self.parking_key = None

    def __enter__(self):
        self.btree.register_cursor(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.btree.deregister_cursor(self)
        return False


class Immutable(Exception):
    """The BTree is immutable."""


class BTree(Generic[KT, ET]):
    """An in-memory BTree with copy-on-write and cursors."""

    def __init__(self, *, t: int = DEFAULT_T, original: Optional["BTree"] = None):
        """Create a BTree.

        If *original* is not ``None``, then the BTree is shallow-cloned from
        *original* using copy-on-write.  Otherwise a new BTree with the specified
        *t* value is created.

        The BTree is not thread-safe.
        """
        # We don't use a reference to ourselves as a creator as we don't want
        # to prevent GC of old btrees.
        self.creator = _Creator()
        self._immutable = False
        self.t: int
        self.root: _Node
        self.size: int
        self.cursors: set[Cursor] = set()
        if original is not None:
            if not original._immutable:
                raise ValueError("original BTree is not immutable")
            self.t = original.t
            self.root = original.root
            self.size = original.size
        else:
            if t < 3:
                raise ValueError("t must be >= 3")
            self.t = t
            self.root = _Node(self.t, self.creator, True)
            self.size = 0

    def make_immutable(self):
        """Make the BTree immutable.

        Attempts to alter the BTree after making it immutable will raise an
        Immutable exception.  This operation cannot be undone.
        """
        if not self._immutable:
            self._immutable = True

    def _check_mutable_and_park(self) -> None:
        if self._immutable:
            raise Immutable
        for cursor in self.cursors:
            cursor.park()

    # Note that we don't use insert() and delete() but rather insert_element() and
    # delete_key() so that BTreeDict can be a proper MutableMapping and supply the
    # rest of the standard mapping API.

    def insert_element(self, elt: ET, in_order: bool = False) -> ET | None:
        """Insert the element into the BTree.

        If *in_order* is ``True``, then extra work will be done to make left siblings
        full, which optimizes storage space when the the elements are inserted in-order
        or close to it.

        Returns the previously existing element at the element's key or ``None``.
        """
        self._check_mutable_and_park()
        cloned = self.root.maybe_cow(self.creator)
        if cloned:
            self.root = cloned
        if self.root.is_maximal():
            old_root = self.root
            self.root = _Node(self.t, self.creator, False)
            self.root.adopt(*old_root.split())
        oelt = self.root.insert_nonfull(elt, in_order)
        if oelt is None:
            # We did not replace, so something was added.
            self.size += 1
        return oelt

    def get_element(self, key: KT) -> ET | None:
        """Get the element matching *key* from the BTree, or return ``None`` if it
        does not exist.
        """
        return self.root.get(key)

    def _delete(self, key: KT, exact: ET | None) -> ET | None:
        self._check_mutable_and_park()
        cloned = self.root.maybe_cow(self.creator)
        if cloned:
            self.root = cloned
        elt = self.root.delete(key, None, exact)
        if elt is not None:
            # We deleted something
            self.size -= 1
            if len(self.root.elts) == 0:
                # The root is now empty.  If there is a child, then collapse this root
                # level and make the child the new root.
                if not self.root.is_leaf:
                    assert len(self.root.children) == 1
                    self.root = self.root.children[0]
        return elt

    def delete_key(self, key: KT) -> ET | None:
        """Delete the element matching *key* from the BTree.

        Returns the matching element or ``None`` if it does not exist.
        """
        return self._delete(key, None)

    def delete_exact(self, element: ET) -> ET | None:
        """Delete *element* from the BTree.

        Returns the matching element or ``None`` if it was not in the BTree.
        """
        delt = self._delete(element.key(), element)
        assert delt is element
        return delt

    def __len__(self):
        return self.size

    def visit_in_order(self, visit: Callable[[ET], None]) -> None:
        """Call *visit*(element) on all elements in the tree in sorted order."""
        self.root.visit_in_order(visit)

    def _visit_preorder_by_node(self, visit: Callable[[_Node], None]) -> None:
        self.root._visit_preorder_by_node(visit)

    def cursor(self) -> Cursor[KT, ET]:
        """Create a cursor."""
        return Cursor(self)

    def register_cursor(self, cursor: Cursor) -> None:
        """Register a cursor for the automatic parking service."""
        self.cursors.add(cursor)

    def deregister_cursor(self, cursor: Cursor) -> None:
        """Deregister a cursor from the automatic parking service."""
        self.cursors.discard(cursor)

    def __copy__(self):
        return self.__class__(original=self)

    def __iter__(self):
        with self.cursor() as cursor:
            while True:
                elt = cursor.next()
                if elt is None:
                    break
                yield elt.key()


VT = TypeVar("VT")  # the type of a value in a BTreeDict


class KV(Element, Generic[KT, VT]):
    """The BTree element type used in a ``BTreeDict``."""

    def __init__(self, key: KT, value: VT):
        self._key = key
        self._value = value

    def key(self) -> KT:
        return self._key

    def value(self) -> VT:
        return self._value

    def __str__(self):  # pragma: no cover
        return f"KV({self._key}, {self._value})"

    def __repr__(self):  # pragma: no cover
        return f"KV({self._key}, {self._value})"


class BTreeDict(Generic[KT, VT], BTree[KT, KV[KT, VT]], MutableMapping[KT, VT]):
    """A MutableMapping implemented with a BTree.

    Unlike a normal Python dict, the BTreeDict may be mutated while iterating.
    """

    def __init__(
        self,
        *,
        t: int = DEFAULT_T,
        original: BTree | None = None,
        in_order: bool = False,
    ):
        super().__init__(t=t, original=original)
        self.in_order = in_order

    def __getitem__(self, key: KT) -> VT:
        elt = self.get_element(key)
        if elt is None:
            raise KeyError
        else:
            return cast(KV, elt).value()

    def __setitem__(self, key: KT, value: VT) -> None:
        elt = KV(key, value)
        self.insert_element(elt, self.in_order)

    def __delitem__(self, key: KT) -> None:
        if self.delete_key(key) is None:
            raise KeyError


class Member(Element, Generic[KT]):
    """The BTree element type used in a ``BTreeSet``."""

    def __init__(self, key: KT):
        self._key = key

    def key(self) -> KT:
        return self._key


class BTreeSet(BTree, Generic[KT], MutableSet[KT]):
    """A MutableSet implemented with a BTree.

    Unlike a normal Python set, the BTreeSet may be mutated while iterating.
    """

    def __init__(
        self,
        *,
        t: int = DEFAULT_T,
        original: BTree | None = None,
        in_order: bool = False,
    ):
        super().__init__(t=t, original=original)
        self.in_order = in_order

    def __contains__(self, key: Any) -> bool:
        return self.get_element(key) is not None

    def add(self, value: KT) -> None:
        elt = Member(value)
        self.insert_element(elt, self.in_order)

    def discard(self, value: KT) -> None:
        self.delete_key(value)
